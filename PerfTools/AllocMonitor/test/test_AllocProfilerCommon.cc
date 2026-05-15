#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

#include "PerfTools/AllocMonitor/plugins/AllocProfilerCommon.h"

namespace cms::perftools::allocMon::profiler {

  // Test accessor: declared as friend in StackNodeData to reach private static functions.
  struct AllocProfilerCommonTestAccess {
    template <typename R1, typename R2>
    static std::size_t computeCommonTopEntries(R1&& r1, R2&& r2) {
      return StackNodeData::computeCommonTopEntries(std::forward<R1>(r1), std::forward<R2>(r2));
    }
  };

}  // namespace cms::perftools::allocMon::profiler

namespace {
  // Mock types replacing std::stacktrace for deterministic, controlled tests.
  // MockEntry simulates std::stacktrace_entry: operator== compares identity (id_),
  // while description() returns a human-readable name. This lets tests exercise the
  // description-fallback path in intersectCommonTopEntries separately from operator==.
  struct MockEntry {
    std::string description_;
    int id_ = 0;
    bool operator==(MockEntry const&) const = default;
    std::string description() const { return description_; }
  };

  // MockTrace simulates std::stacktrace: frames_[0] is the innermost frame,
  // frames_.back() is the outermost frame, matching std::stacktrace index layout.
  struct MockTrace {
    std::vector<MockEntry> frames_;
    std::size_t size() const { return frames_.size(); }
    MockEntry const& operator[](std::size_t i) const { return frames_[i]; }
  };

  // Build a MockTrace from a list of frame names, innermost first.
  // Each entry gets a unique id equal to its position, so two traces built from
  // the same names will be operator==-equal entry-by-entry.
  MockTrace makeTrace(std::initializer_list<std::string_view> frameNames) {
    MockTrace t;
    int id = 0;
    for (auto const& name : frameNames) {
      t.frames_.push_back({std::string(name), id++});
    }
    return t;
  }
}  // namespace

using TestAccess = cms::perftools::allocMon::profiler::AllocProfilerCommonTestAccess;

TEST_CASE("computeCommonTopEntries", "[AllocProfilerCommon]") {
  // ----------------------------------------------------------------------------
  // Edge cases: empty / single-trace inputs always return 0.

  SECTION("both ranges empty") {
    std::vector<MockTrace> empty;
    CHECK(TestAccess::computeCommonTopEntries(empty, empty) == 0);
  }

  SECTION("single trace in first range, second empty") {
    std::vector<MockTrace> r1 = {makeTrace({"inner", "outer"})};
    std::vector<MockTrace> empty;
    CHECK(TestAccess::computeCommonTopEntries(r1, empty) == 0);
  }

  SECTION("single trace in second range, first empty") {
    std::vector<MockTrace> empty;
    std::vector<MockTrace> r2 = {makeTrace({"inner", "outer"})};
    CHECK(TestAccess::computeCommonTopEntries(empty, r2) == 0);
  }

  // ----------------------------------------------------------------------------
  // Identical traces: every frame matches, so the full trace length is returned.

  SECTION("two identical traces, one per range") {
    auto const t = makeTrace({"inner", "middle", "outer"});
    std::vector<MockTrace> r1 = {t}, r2 = {t};
    CHECK(TestAccess::computeCommonTopEntries(r1, r2) == 3);
  }

  SECTION("two identical traces in first range, second empty") {
    auto const t = makeTrace({"inner", "middle", "outer"});
    std::vector<MockTrace> r1 = {t, t};
    std::vector<MockTrace> empty;
    CHECK(TestAccess::computeCommonTopEntries(r1, empty) == 3);
  }

  SECTION("multiple identical traces across both ranges") {
    auto const t = makeTrace({"inner", "middle", "outer"});
    std::vector<MockTrace> r1 = {t, t, t}, r2 = {t, t};
    CHECK(TestAccess::computeCommonTopEntries(r1, r2) == 3);
  }

  // ----------------------------------------------------------------------------
  // Partial match: traces share only a subset of their outermost frames.

  SECTION("full trace vs. trace with one inner frame removed") {
    // full:    [inner, middle, outer]  (3 frames)
    // partial: [middle, outer]         (inner removed)
    // The 2 outermost frames match.
    auto const full = makeTrace({"inner", "middle", "outer"});
    auto const partial = makeTrace({"middle", "outer"});
    std::vector<MockTrace> r1 = {full}, r2 = {partial};
    CHECK(TestAccess::computeCommonTopEntries(r1, r2) == 2);
  }

  SECTION("full trace vs. single outermost frame") {
    auto const full = makeTrace({"inner", "middle", "outer"});
    auto const outer = makeTrace({"outer"});
    std::vector<MockTrace> r1 = {full}, r2 = {outer};
    CHECK(TestAccess::computeCommonTopEntries(r1, r2) == 1);
  }

  SECTION("traces sharing only the outermost frame") {
    auto const t1 = makeTrace({"funcA", "common_outer"});
    auto const t2 = makeTrace({"funcB", "common_outer"});
    std::vector<MockTrace> r1 = {t1}, r2 = {t2};
    CHECK(TestAccess::computeCommonTopEntries(r1, r2) == 1);
  }

  SECTION("traces sharing two outermost frames out of three") {
    auto const t1 = makeTrace({"unique_A", "shared_middle", "shared_outer"});
    auto const t2 = makeTrace({"unique_B", "shared_middle", "shared_outer"});
    std::vector<MockTrace> r1 = {t1}, r2 = {t2};
    CHECK(TestAccess::computeCommonTopEntries(r1, r2) == 2);
  }

  SECTION("no common frames") {
    auto const t1 = makeTrace({"A", "B", "C"});
    auto const t2 = makeTrace({"D", "E", "F"});
    std::vector<MockTrace> r1 = {t1}, r2 = {t2};
    CHECK(TestAccess::computeCommonTopEntries(r1, r2) == 0);
  }

  // ----------------------------------------------------------------------------
  // Multiple traces in a range reduce the common count to the intersection.

  SECTION("multiple traces, common set is the smallest overlap") {
    // t1 and t2 share 2 outermost frames; t1 and t3 share only 1.
    // Expected: the intersection is 1.
    auto const t1 = makeTrace({"u1", "shared_mid", "shared_top"});
    auto const t2 = makeTrace({"u2", "shared_mid", "shared_top"});
    auto const t3 = makeTrace({"u3", "other_mid", "shared_top"});
    std::vector<MockTrace> r1 = {t1, t2}, r2 = {t3};
    CHECK(TestAccess::computeCommonTopEntries(r1, r2) == 1);
  }

  // ----------------------------------------------------------------------------
  // Description-fallback: frames where operator== fails but description() matches
  // are still counted as common (simulating inlined frames in real stacktraces).

  SECTION("description-only match counts as common") {
    // The outer frames have the same description but different ids, so operator==
    // returns false and the description fallback is exercised.
    MockTrace t_ref{{MockEntry{"inner_A", 0}, MockEntry{"outer", 100}}};
    MockTrace t_tr{{MockEntry{"inner_B", 1}, MockEntry{"outer", 200}}};
    std::vector<MockTrace> r1 = {t_ref}, r2 = {t_tr};
    // outer: op== false, description matches → common (k=1)
    // inner: op== false, descriptions differ  → stop
    CHECK(TestAccess::computeCommonTopEntries(r1, r2) == 1);
  }

  SECTION("description-only match for all frames") {
    // All entries have matching descriptions but distinct ids.
    MockTrace t1{{MockEntry{"A", 10}, MockEntry{"B", 20}, MockEntry{"C", 30}}};
    MockTrace t2{{MockEntry{"A", 40}, MockEntry{"B", 50}, MockEntry{"C", 60}}};
    std::vector<MockTrace> r1 = {t1}, r2 = {t2};
    CHECK(TestAccess::computeCommonTopEntries(r1, r2) == 3);
  }
}
