#include <algorithm>
#include <cassert>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <vector>

#include <catch2/catch_all.hpp>

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/TestHandle.h"
#include "DataFormats/Provenance/interface/ProductID.h"

using namespace edm;

//------------------------------------------------------
// This is a sample VALUE class, almost the simplest possible.
//------------------------------------------------------

struct Empty {};

template <typename BASE>
class ValueWithKeyT : public BASE {
public:
  // VALUES must be default constructible
  ValueWithKeyT() : d_(0.0) {}

  // VALUES must be destructible
  ~ValueWithKeyT() = default;

  // If VALUES provide a sort_key() const method, it will be used
  // to generate keys once, instead of calling operator<.
  double sort_key() const { return d_; }

  // VALUES must be LessThanComparable
  bool operator<(ValueWithKeyT const& other) const { return sort_key() < other.sort_key(); }

  // The stuff below is all implementation detail, and not
  // required by the concept VALUE.

  // This constructor is used for testing; it is not required by the
  // concept VALUE.
  explicit ValueWithKeyT(double d) : d_(d) {}

  // This access function is used for testing; it is not required by
  // the concept VALUE.
  double val() const { return d_; }

private:
  double d_;
};

typedef edm::DoNotSortUponInsertion DNS;

template <>
double ValueWithKeyT<DNS>::sort_key() const {
  throw std::logic_error("You can't sort me!");
}

typedef ValueWithKeyT<Empty> Value;
//typedef ValueWithKeyT<DNS> Value; // NoSort;

//------------------------------------------------------
// The stream insertion operator is not required; it is used here
// for diagnostic output.
//
// It is inline to avoid multiple definition problems. Note that this
// cc file is *not* a compilation unit; it is actually an include
// file, which the build system combines with others to create a
// compilation unit.
//
//------------------------------------------------------

template <typename BASE>
std::ostream& operator<<(std::ostream& os, ValueWithKeyT<BASE> const& v) {
  os << " val: " << v.val();
  return os;
}

typedef edm::DetSetVector<Value> coll_type;
typedef coll_type::detset detset;

void check_outer_collection_order(coll_type const& c) {
  if (c.size() < 2)
    return;
  coll_type::const_iterator i = c.begin();
  coll_type::const_iterator e = c.end();
  // Invariant: sequence from prev to i is correctly ordered
  coll_type::const_iterator prev(i);
  ++i;
  for (; i != e; ++i, ++prev) {
    // We don't use CPPUNIT_ASSERT because it gives us grossly
    // insufficient context if a failure occurs.
    REQUIRE(prev->id < i->id);
  }
}

void check_inner_collection_order(detset const& d) {
  if (d.data.size() < 2)
    return;
  detset::const_iterator i = d.data.begin();
  detset::const_iterator e = d.data.end();
  // Invariant: sequence from prev to i is correctly ordered
  detset::const_iterator prev(i);
  ++i;
  for (; i != e; ++i, ++prev) {
    // We don't use CPPUNIT_ASSERT because it gives us grossly
    // insufficient context if a failure occurs.
    //
    // We don't check that *prev < *i because they might be equal.
    // We don't want to require an op<= or op==.
    REQUIRE(!(*i < *prev));
  }
}

void printDetSet(detset const& ds, std::ostream& os) {
  os << "size: " << ds.data.size() << '\n' << "values: ";
  std::copy(ds.data.begin(), ds.data.end(), std::ostream_iterator<detset::value_type>(os, " "));
}

void sanity_check(coll_type const& c) {
  check_outer_collection_order(c);
  for (coll_type::const_iterator i = c.begin(), e = c.end(); i != e; ++i) {
    //       printDetSet(*i, std::cerr);
    //       std::cerr << '\n';
    check_inner_collection_order(*i);
  }
}

void check_ids(coll_type const& c) {
  // Long way to get all ids...
  std::vector<det_id_type> all_ids;
  for (coll_type::const_iterator i = c.begin(), e = c.end(); i != e; ++i) {
    all_ids.push_back(i->id);
  }
  REQUIRE(c.size() == all_ids.size());

  std::vector<det_id_type> nice_ids;
  c.getIds(nice_ids);
  REQUIRE(all_ids == nice_ids);
}

namespace {
  template <typename T>
  struct DSVGetter : edm::EDProductGetter {
    DSVGetter() : edm::EDProductGetter(), prod_(nullptr) {}
    WrapperBase const* getIt(ProductID const&) const override { return prod_; }

    std::optional<std::tuple<edm::WrapperBase const*, unsigned int>> getThinnedProduct(ProductID const&,
                                                                                       unsigned int) const override {
      return std::nullopt;
    }

    void getThinnedProducts(ProductID const& pid,
                            std::vector<WrapperBase const*>& wrappers,
                            std::vector<unsigned int>& keys) const override {}

    edm::OptionalThinnedKey getThinnedKeyFrom(ProductID const&, unsigned int, ProductID const&) const override {
      return std::monostate{};
    }

    unsigned int transitionIndex_() const override { return 0U; }

    edm::Wrapper<T> const* prod_;
  };
}  // namespace

TEST_CASE("test DetSetVector with sort_key", "[DetSetVector]") {
  SECTION("detsetTest") {
    //std::cerr << "\nStart DetSetVector_t detsetTest()\n";
    detset d;
    Value v1(1.1);
    Value v2(2.2);
    d.id = edm::det_id_type(3);
    d.data.push_back(v1);
    d.data.push_back(v2);
    edm::detail::sort_by_key(d.data.begin(), d.data.end(), [](const Value& v) { return v.sort_key(); });
    check_inner_collection_order(d);
    //std::cerr << "\nEnd DetSetVector_t detsetTest()\n";
  }

  SECTION("refTest") {
    coll_type c;
    detset d3;
    Value v1(1.1);
    Value v2(2.2);
    d3.id = edm::det_id_type(3);
    d3.data.push_back(v1);
    d3.data.push_back(v2);
    c.insert(d3);
    detset d1;
    Value v1a(4.1);
    Value v2a(3.2);
    d1.id = edm::det_id_type(1);
    d1.data.push_back(v1a);
    d1.data.push_back(v2a);
    c.insert(d1);
    c.post_insert();

    auto pC = std::make_unique<coll_type>(c);
    edm::Wrapper<coll_type> wrapper(std::move(pC));
    DSVGetter<coll_type> theGetter;
    theGetter.prod_ = &wrapper;

    typedef edm::Ref<coll_type, detset> RefDetSet;
    typedef edm::Ref<coll_type, Value> RefDet;

    {
      RefDetSet refSet(edm::ProductID(1, 1), 0, &theGetter);
      REQUIRE((!(d1 < *refSet) && !(*refSet < d1)));
    }
    {
      RefDetSet refSet(edm::ProductID(1, 1), 1, &theGetter);
      REQUIRE((!(d3 < *refSet) && !(*refSet < d3)));
    }
    {
      RefDet refDet(edm::ProductID(1, 1), RefDet::key_type(3, 0), &theGetter);
      REQUIRE((!(v1 < *refDet) && !(*refDet < v1)));
    }

    {
      TestHandle<coll_type> pc2(&c, ProductID(1, 1));
      RefDet refDet = makeRefToDetSetVector(pc2, det_id_type(3), c[3].data.begin());
      REQUIRE((!(v1 < *refDet) && !(*refDet < v1)));
    }

    using namespace Catch::Matchers;
    SECTION("bad detid") {
      TestHandle<coll_type> pc2(&c, ProductID(1, 1));
      REQUIRE_THROWS_MATCHES(
          makeRefToDetSetVector(pc2, det_id_type(12), c[3].data.begin()),
          edm::Exception,
          Predicate<edm::Exception>(
              [](auto const& iExcept) { return iExcept.categoryCode() == edm::errors::InvalidReference; },
              "Exception has wrong category"));
    }

    SECTION("bad iterator") {
      TestHandle<coll_type> pc2(&c, ProductID(1, 1));
      REQUIRE_THROWS_MATCHES(
          makeRefToDetSetVector(pc2, det_id_type(1), c[3].data.begin()),
          edm::Exception,
          Predicate<edm::Exception>([](auto const& x) { return x.categoryCode() == edm::errors::InvalidReference; },
                                    "Exception has wrong category"));
    }
  }

  SECTION("work") {
    coll_type c1;
    c1.post_insert();
    sanity_check(c1);
    REQUIRE(c1.size() == 0);
    REQUIRE(c1.empty());

    coll_type c2(c1);
    REQUIRE(c2.size() == c1.size());
    sanity_check(c2);
    coll_type c;
    sanity_check(c);
    {
      detset d;
      Value v1(1.1);
      Value v2(2.2);
      d.id = edm::det_id_type(3);
      d.data.push_back(v1);
      d.data.push_back(v2);
      c.insert(d);
      c.post_insert();
    }
    sanity_check(c);
    REQUIRE(c.size() == 1);
    {
      detset d;
      Value v1(4.1);
      Value v2(3.2);
      d.id = edm::det_id_type(1);
      d.data.push_back(v1);
      d.data.push_back(v2);
      c.insert(d);
      c.post_insert();
    }
    sanity_check(c);
    REQUIRE(c.size() == 2);

    {
      detset d;
      Value v1(1.1);
      Value v2(1.2);
      Value v3(2.2);
      d.id = edm::det_id_type(10);
      d.data.push_back(v3);
      d.data.push_back(v2);
      d.data.push_back(v1);
      c.insert(d);
      c.post_insert();
    }
    sanity_check(c);
    REQUIRE(c.size() == 3);

    coll_type another;
    c.swap(another);
    REQUIRE(c.empty());
    REQUIRE(another.size() == 3);
    sanity_check(c);
    sanity_check(another);

    c.swap(another);
    REQUIRE(c.size() == 3);

    {
      // We should not find anything with ID=11
      coll_type::iterator i = c.find(edm::det_id_type(11));
      REQUIRE(i == c.end());

      coll_type::const_iterator ci = static_cast<coll_type const&>(c).find(edm::det_id_type(11));
      REQUIRE(ci == c.end());
    }
    {
      // We should find  ID=10
      coll_type::iterator i = c.find(edm::det_id_type(10));
      REQUIRE(i != c.end());
      REQUIRE(i->id == 10);
      REQUIRE(i->data.size() == 3);
    }
    using namespace Catch::Matchers;
    {
      // We should not find ID=100; op[] should throw.
      SECTION("op[] should throw") {
        REQUIRE_THROWS_MATCHES(
            c[edm::det_id_type(100)],
            edm::Exception,
            Predicate<edm::Exception>([](auto const& x) { return x.categoryCode() == edm::errors::InvalidReference; },
                                      "Exception has wrong category"));
      }
    }

    {
      // We should not find ID=100; op[] should throw.
      SECTION("coll_type op[] should throw") {
        REQUIRE_THROWS_MATCHES(
            static_cast<coll_type const&>(c)[edm::det_id_type(100)],
            edm::Exception,
            Predicate<edm::Exception>([](auto const& x) { return x.categoryCode() == edm::errors::InvalidReference; },
                                      "Exception has wrong category"));
      }
    }
    {
      // We should find id = 3
      coll_type const& rc = c;
      coll_type::const_reference r = rc[3];
      REQUIRE(r.id == edm::det_id_type(3));
      REQUIRE(r.data.size() == 2);

      coll_type::reference r2 = c[3];
      REQUIRE(r2.id == edm::det_id_type(3));
      REQUIRE(r2.data.size() == 2);
    }

    {
      // We should not find id = 17, but a new empty DetSet should be
      // inserted, carrying the correct DetId.
      coll_type::size_type oldsize = c.size();
      coll_type::reference r = c.find_or_insert(edm::det_id_type(17));
      coll_type::size_type newsize = c.size();
      REQUIRE(newsize > oldsize);
      REQUIRE(newsize == (oldsize + 1));
      REQUIRE(r.id == edm::det_id_type(17));
      REQUIRE(r.data.size() == 0);
      r.data.push_back(Value(10.1));
      r.data.push_back(Value(9.1));
      r.data.push_back(Value(4.0));
      r.data.push_back(Value(4.0));
      c.post_insert();
      sanity_check(c);
    }
    {
      // Make sure we can swap in a vector.
      unsigned int const numDetSets = 20;
      unsigned int const detSetSize = 14;
      std::vector<detset> v;
      for (unsigned int i = 0; i < numDetSets; ++i) {
        detset d(i);
        for (unsigned int j = 0; j < detSetSize; ++j) {
          d.data.push_back(Value(100 * i + 1.0 / j));
        }
        v.push_back(d);
      }
      REQUIRE(v.size() == numDetSets);
      coll_type c3(v);
      c3.post_insert();
      REQUIRE(v.size() == 0);
      REQUIRE(c3.size() == numDetSets);
      sanity_check(c3);

      coll_type c4;
      c4 = c3;
      REQUIRE(c4.size() == numDetSets);
      sanity_check(c3);
      sanity_check(c4);

      check_ids(c3);
    }
  }
}
