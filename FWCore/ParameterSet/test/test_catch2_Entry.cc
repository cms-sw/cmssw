#include "catch2/catch_all.hpp"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <string_view>
#include <vector>

namespace {
  // build an Entry from its type name and encoded value, and check that it matches the original one
  void checkRoundTrip(edm::Entry const& entry) {
    std::string const coded = entry.toString();
    std::string_view const value = std::string_view(coded).substr(3, coded.size() - 4);
    edm::Entry const copy(entry.name(), entry.typeName(), value, entry.isTracked());
    CHECK(copy.isTracked() == entry.isTracked());
    CHECK(copy.typeName() == entry.typeName());
    CHECK(copy.toString() == coded);
  }

  template <typename T>
  void checkRoundTrip(T const& value) {
    INFO("tracked");
    checkRoundTrip(edm::Entry("value", value, true));
    INFO("untracked");
    checkRoundTrip(edm::Entry("value", value, false));
  }
}  // namespace

TEST_CASE("Entry from type name and encoded value", "[Entry]") {
  SECTION("scalar types") {
    checkRoundTrip(true);
    checkRoundTrip(-42);
    checkRoundTrip(42u);
    checkRoundTrip(-42ll);
    checkRoundTrip(42ull);
    checkRoundTrip(3.14);
    checkRoundTrip(3.14f);
    checkRoundTrip(std::string("hello world"));
    checkRoundTrip(edm::InputTag("module", "instance", "process"));
    checkRoundTrip(edm::ESInputTag("module", "data"));
    checkRoundTrip(edm::EventID(1, 2, 3));
    checkRoundTrip(edm::LuminosityBlockID(1, 2));
    checkRoundTrip(edm::EventRange(1, 2, 3, 4, 5, 6));
    checkRoundTrip(edm::LuminosityBlockRange(1, 2, 3, 4));
  }

  SECTION("vector types") {
    checkRoundTrip(std::vector<int>{-1, 0, 1});
    checkRoundTrip(std::vector<unsigned>{0u, 1u, 2u});
    checkRoundTrip(std::vector<long long>{-1ll, 0ll, 1ll});
    checkRoundTrip(std::vector<unsigned long long>{0ull, 1ull, 2ull});
    checkRoundTrip(std::vector<double>{-1.5, 0., 1.5});
    checkRoundTrip(std::vector<float>{-1.5f, 0.f, 1.5f});
    checkRoundTrip(std::vector<std::string>{"hello", "world"});
    checkRoundTrip(std::vector<edm::InputTag>{edm::InputTag("a"), edm::InputTag("b", "c")});
    checkRoundTrip(std::vector<edm::ESInputTag>{edm::ESInputTag("a", "b"), edm::ESInputTag("", "c")});
    checkRoundTrip(std::vector<edm::EventID>{edm::EventID(1, 2, 3), edm::EventID(4, 5, 6)});
    checkRoundTrip(std::vector<edm::LuminosityBlockID>{edm::LuminosityBlockID(1, 2), edm::LuminosityBlockID(3, 4)});
    checkRoundTrip(std::vector<edm::EventRange>{edm::EventRange(1, 2, 3, 4, 5, 6)});
    checkRoundTrip(std::vector<edm::LuminosityBlockRange>{edm::LuminosityBlockRange(1, 2, 3, 4)});
  }

  SECTION("vector of encoded values") {
    // integers are encoded with an explicit sign
    edm::Entry const tracked("value", "vint32", std::vector<std::string>{"-1", "+0", "+1"}, true);
    CHECK(tracked.isTracked());
    CHECK(tracked.getVInt32() == std::vector<int>{-1, 0, 1});

    edm::Entry const untracked("value", "vint32", std::vector<std::string>{"-1", "+0", "+1"}, false);
    CHECK(not untracked.isTracked());
    CHECK(untracked.getVInt32() == std::vector<int>{-1, 0, 1});
  }

  SECTION("invalid type name") { CHECK_THROWS_AS(edm::Entry("value", "invalid", "1", true), edm::Exception); }
}
