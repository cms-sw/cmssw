#include "catch2/catch_all.hpp"

#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetAlgorithm.h"
#include "DataFormats/Common/interface/DetSet2RangeMap.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <vector>

namespace {
  struct B {
    virtual ~B() {}
    virtual B *clone() const = 0;
  };

  struct T : public B {
    T(float iv = 0) : v(iv) {}
    float v;
    bool operator==(T t) const { return v == t.v; }

    T *clone() const final { return new T(*this); }
  };

  bool operator==(T const &t, B const &b) {
    T const *p = dynamic_cast<T const *>(&b);
    return p && p->v == t.v;
  }

  bool operator==(B const &b, T const &t) { return t == b; }
}  // namespace
typedef edmNew::DetSetVector<T> DSTV;
typedef edmNew::DetSet<T> DST;
typedef edmNew::det_id_type det_id_type;
typedef DSTV::FastFiller FF;
typedef DSTV::TSFastFiller TSFF;

class TestDetSet {
public:
  static auto &data(DSTV &detsets) { return detsets.m_data; }
  static auto &item(FF &ff) { return ff.m_item; }
  static auto &item(TSFF &ff) { return ff.m_item; }
};
namespace {
  struct VerifyIter {
    VerifyIter(std::vector<DSTV::data_type> *sv, unsigned int in = 1, int iincr = 1) : n(in), incr(iincr), sv_(sv) {}

    void operator()(DST const &df) {
      if (df.id() > 1000) {
        REQUIRE(df.size() == 0);
        return;
      }
      REQUIRE(df.id() == 20 + n);
      REQUIRE(df.size() == n);
      std::vector<DST::data_type> v1(n);
      std::vector<DST::data_type> v2(n);
      std::copy(df.begin(), df.end(), v2.begin());
      std::copy(sv_->begin(), sv_->begin() + n, v1.begin());
      REQUIRE(v1 == v2);
      n += incr;
    }

    unsigned int n;
    int incr;
    std::vector<DSTV::data_type> *sv_;
  };

  struct cmp10 {
    bool operator()(DSTV::id_type i1, DSTV::id_type i2) const { return i1 / 10 < i2 / 10; }
  };

  std::pair<unsigned int, cmp10> acc(unsigned int i) { return std::make_pair(i * 10, cmp10()); }

  struct VerifyAlgos {
    VerifyAlgos(std::vector<DSTV::data_type const *> &iv) : n(0), v(iv) {}

    void operator()(DSTV::data_type const &d) {
      REQUIRE(d == *v[n]);
      REQUIRE(&d == v[n]);
      ++n;
    }

    int n;
    std::vector<DSTV::data_type const *> const &v;
  };

  struct Getter final : public DSTV::Getter {
    Getter(std::vector<DSTV::data_type> *sv) : ntot(0), sv_(*sv) {}

    void fill(TSFF &ff) const override {
      aborted = false;
      try {
        const int n = ff.id() - 20;
        REQUIRE(n > 0);
        ff.resize(n);
        int nCopied = n;
        if (static_cast<size_t>(n) > sv_.size()) {
          nCopied = sv_.size();
        }
        std::copy(sv_.begin(), sv_.begin() + nCopied, ff.begin());
        if (ff.full()) {
          ff.abort();
          aborted = true;
        } else {
          ntot += n;
        }
      } catch (edmNew::CapacityExaustedException const &) {
        REQUIRE(false);
      }
    }

    mutable unsigned int ntot;
    mutable bool aborted = false;
    std::vector<DSTV::data_type> &sv_;
  };
}  // namespace

TEST_CASE("DetSetNew", "[DetSetNew]") {
  std::vector<DSTV::data_type> sv(10);
  DSTV::data_type v[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  std::copy(v, v + 10, sv.begin());

  SECTION("constructor") {
    DSTV detsets(2);
    REQUIRE(!detsets.onDemand());
    REQUIRE(detsets.subdetId() == 2);
    REQUIRE(detsets.size() == 0);
    REQUIRE(detsets.empty());
    REQUIRE(detsets.dataSize() == 0);
    detsets.resize(3, 10);
    REQUIRE(detsets.size() == 3);
    REQUIRE(detsets.dataSize() == 10);
    REQUIRE(!detsets.empty());
    // follow is nonsense still valid construct... (maybe it shall throw...)
    // now assert!
    /*
    DST df(detsets,detsets.item(1));
    CPPUNIT_ASSERT(df.id()==0);
    CPPUNIT_ASSERT(df.size()==0);
    CPPUNIT_ASSERT(df.data()+1==&TestDetSet::data(detsets).front());
    df.set(detsets,detsets.item(2));
    CPPUNIT_ASSERT(df.size()==0);
    CPPUNIT_ASSERT(df.data()+1==&TestDetSet::data(detsets).front());
    */

    SECTION("swap") {
      DSTV detsets2(3);
      detsets.swap(detsets2);
      REQUIRE(detsets.subdetId() == 3);
      REQUIRE(detsets.size() == 0);
      REQUIRE(detsets.dataSize() == 0);
      REQUIRE(detsets2.subdetId() == 2);
      REQUIRE(detsets2.size() == 3);
      REQUIRE(detsets2.dataSize() == 10);
      REQUIRE(!detsets2.onDemand());
    }
    SECTION("move") {
      DSTV detsets3(4);
      detsets = std::move(detsets3);
      REQUIRE(detsets.subdetId() == 4);
      REQUIRE(detsets.size() == 0);
      REQUIRE(detsets.dataSize() == 0);
      DSTV detsets5(std::move(detsets));
      REQUIRE(detsets5.subdetId() == 4);
      REQUIRE(detsets5.size() == 0);
      REQUIRE(detsets5.dataSize() == 0);
      SECTION("copy") {
        DSTV detsets4(detsets5);
        REQUIRE(detsets4.subdetId() == 4);
        REQUIRE(detsets4.size() == 0);
        REQUIRE(detsets4.dataSize() == 0);
      }
    }
  }

  SECTION("inserting") {
    DSTV detsets(2);
    unsigned int ntot = 0;
    for (unsigned int n = 1; n < 5; ++n) {
      ntot += n;
      unsigned int id = 20 + n;
      DST df = detsets.insert(id, n);
      REQUIRE(detsets.size() == n);
      REQUIRE(detsets.dataSize() == ntot);
      REQUIRE(detsets.detsetSize(n - 1) == n);
      REQUIRE(df.size() == n);
      REQUIRE(df.id() == id);

      std::copy(sv.begin(), sv.begin() + n, df.begin());

      DSTV detsets2(detsets);
      REQUIRE(detsets2.size() == n);
      REQUIRE(detsets2.dataSize() == ntot);
      REQUIRE(detsets2.detsetSize(n - 1) == n);

      std::vector<DST::data_type> v1(n);
      std::vector<DST::data_type> v2(n);
      std::vector<DST::data_type> v3(n);
      std::copy(TestDetSet::data(detsets).begin() + ntot - n, TestDetSet::data(detsets).begin() + ntot, v2.begin());
      std::copy(TestDetSet::data(detsets2).begin() + ntot - n, TestDetSet::data(detsets2).begin() + ntot, v3.begin());
      std::copy(sv.begin(), sv.begin() + n, v1.begin());
      REQUIRE(v1 == v2);
      REQUIRE(v1 == v3);
    }

    // test error conditions
    REQUIRE_THROWS_MATCHES(
        detsets.insert(22, 6), edm::Exception, Catch::Matchers::Predicate<edm::Exception>([](edm::Exception const &e) {
          return e.categoryCode() == edm::errors::InvalidReference;
        }));
  }

  SECTION("filling") {
    DSTV detsets(2);
    unsigned int ntot = 0;
    for (unsigned int n = 1; n < 5; ++n) {
      unsigned int id = 20 + n;
      FF ff(detsets, id);
      REQUIRE(detsets.size() == n);
      REQUIRE(detsets.dataSize() == ntot);
      REQUIRE(detsets.detsetSize(n - 1) == 0);
      REQUIRE(TestDetSet::item(ff).offset == int(detsets.dataSize()));
      REQUIRE(TestDetSet::item(ff).size == 0);
      REQUIRE(TestDetSet::item(ff).id == id);
      ntot += 1;
      ff.push_back(3.14);
      REQUIRE(detsets.dataSize() == ntot);
      REQUIRE(detsets.detsetSize(n - 1) == 1);
      REQUIRE(TestDetSet::data(detsets).back().v == 3.14f);
      REQUIRE(TestDetSet::item(ff).offset == int(detsets.dataSize()) - 1);
      REQUIRE(TestDetSet::item(ff).size == 1);
      ntot += n - 1;
      ff.resize(n);
      REQUIRE(detsets.dataSize() == ntot);
      REQUIRE(detsets.detsetSize(n - 1) == n);
      REQUIRE(TestDetSet::item(ff).offset == int(detsets.dataSize() - n));
      REQUIRE(TestDetSet::item(ff).size == n);

      std::copy(sv.begin(), sv.begin() + n, ff.begin());

      std::vector<DST::data_type> v1(n);
      std::vector<DST::data_type> v2(n);
      std::copy(TestDetSet::data(detsets).begin() + ntot - n, TestDetSet::data(detsets).begin() + ntot, v2.begin());
      std::copy(sv.begin(), sv.begin() + n, v1.begin());
      REQUIRE(v1 == v2);
    }

    // test abort and empty
    {
      FF ff1(detsets, 30);
      REQUIRE(detsets.size() == 5);
      REQUIRE(detsets.exists(30));
      REQUIRE(detsets.isValid(30));
    }
    REQUIRE(detsets.size() == 4);
    REQUIRE(!detsets.exists(30));
    {
      FF ff1(detsets, 30, true);
      REQUIRE(detsets.size() == 5);
      REQUIRE(detsets.exists(30));
      REQUIRE(detsets.isValid(30));
    }
    REQUIRE(detsets.size() == 5);
    REQUIRE(detsets.exists(30));

    {
      DSTV detsets2(detsets);
      REQUIRE(detsets2.size() == 5);
      REQUIRE(detsets2.exists(30));
    }

    {
      unsigned int cs = detsets.dataSize();
      FF ff1(detsets, 31);
      ff1.resize(4);
      REQUIRE(detsets.size() == 6);
      REQUIRE(detsets.dataSize() == cs + 4);
      REQUIRE(detsets.exists(31));
      REQUIRE(detsets.isValid(31));
      ff1.abort();
      REQUIRE(detsets.size() == 5);
      REQUIRE(detsets.dataSize() == cs);
      REQUIRE(!detsets.exists(31));
    }
    REQUIRE(detsets.size() == 5);
    REQUIRE(!detsets.exists(31));
    { FF ff1(detsets, 32, true); }
    REQUIRE(detsets.size() == 6);

    DSTV detsets2(detsets);
    REQUIRE(detsets2.size() == 6);

    detsets.clean();
    REQUIRE(detsets.size() == 4);
    REQUIRE(!detsets.exists(30));
    REQUIRE(!detsets.exists(32));

    REQUIRE(detsets2.size() == 6);
    REQUIRE(detsets2.exists(30));
    REQUIRE(detsets2.exists(32));
    detsets2.clean();
    REQUIRE(detsets2.size() == 4);
    REQUIRE(!detsets2.exists(30));
    REQUIRE(!detsets2.exists(32));

    // test error conditions
    REQUIRE_THROWS_MATCHES(
        FF(detsets, 22), edm::Exception, Catch::Matchers::Predicate<edm::Exception>([](edm::Exception const &e) {
          return e.categoryCode() == edm::errors::InvalidReference;
        }));
    REQUIRE_THROWS_MATCHES(
        [](DSTV &d) {
          FF ff1(d, 44);
          FF ff2(d, 45);
        }(detsets),
        edm::Exception,
        Catch::Matchers::Predicate<edm::Exception>(
            [](edm::Exception const &e) { return e.categoryCode() == edm::errors::LogicError; }));
  }

  SECTION("filling using TSFastFiller") {
    DSTV detsets(2);
    detsets.reserve(5, 100);
    unsigned int ntot = 0;
    for (unsigned int n = 1; n < 5; ++n) {
      unsigned int id = 20 + n;
      {
        TSFF ff(detsets, id);
        REQUIRE(detsets.size() == n);
        REQUIRE(detsets.dataSize() == ntot);
        REQUIRE(TestDetSet::item(ff).size == 0);
        REQUIRE(ff.id() == id);
        ntot += 1;
        ff.push_back(3.14);
        REQUIRE(detsets.dataSize() == ntot - 1);
        REQUIRE(TestDetSet::item(ff).size == 0);
        REQUIRE(ff.size() == 1);
        ntot += n - 1;
        ff.resize(n);
        REQUIRE(ff.size() == n);
        std::copy(sv.begin(), sv.begin() + n, ff.begin());
      }
      REQUIRE(detsets.size() == n);
      REQUIRE(detsets.dataSize() == ntot);

      std::vector<DST::data_type> v1(n);
      std::vector<DST::data_type> v2(n);
      std::copy(TestDetSet::data(detsets).begin() + ntot - n, TestDetSet::data(detsets).begin() + ntot, v2.begin());
      std::copy(sv.begin(), sv.begin() + n, v1.begin());
      REQUIRE(v1 == v2);
    }

    // test abort and empty
    {
      TSFF ff1(detsets, 30);
      REQUIRE(detsets.exists(30));
    }
    REQUIRE(detsets.size() == 5);
    REQUIRE(detsets.exists(30));
    detsets.clean();
    REQUIRE(detsets.size() == 4);
    REQUIRE(!detsets.exists(30));
    unsigned int cs = detsets.dataSize();
    SECTION("abort") {
      TSFF ff1(detsets, 31);
      ff1.resize(4);
      ff1.abort();
      REQUIRE(detsets.size() == 5);
      REQUIRE(detsets.dataSize() == cs);
      SECTION("clean") {
        REQUIRE(detsets.exists(31));
        detsets.clean();
        REQUIRE(detsets.size() == 4);
        REQUIRE(!detsets.exists(31));
      }
    }
    SECTION("error conditions") {
      REQUIRE_THROWS_MATCHES(
          TSFF(detsets, 22), edm::Exception, Catch::Matchers::Predicate<edm::Exception>([](edm::Exception const &e) {
            return e.categoryCode() == edm::errors::InvalidReference;
          }));
      REQUIRE_THROWS_MATCHES(
          [](DSTV &d) {
            TSFF ff1(d, 44);
            TSFF ff2(d, 45);
          }(detsets),
          edm::Exception,
          Catch::Matchers::Predicate<edm::Exception>(
              [](edm::Exception const &e) { return e.categoryCode() == edm::errors::LogicError; }));
    }
  }

  SECTION("iterator") {
    DSTV detsets(2);
    for (unsigned int n = 1; n < 5; ++n) {
      unsigned int id = 20 + n;
      FF ff(detsets, id);
      ff.resize(n);
      std::copy(sv.begin(), sv.begin() + n, ff.begin());
    }
    REQUIRE(std::for_each(detsets.begin(true), detsets.end(true), VerifyIter(&sv)).n == 5);
    {
      FF ff(detsets, 31);
      ff.resize(2);
      std::copy(sv.begin(), sv.begin() + 2, ff.begin());
    }
    {
      FF ff(detsets, 11);
      ff.resize(2);
      std::copy(sv.begin(), sv.begin() + 2, ff.begin());
    }
    {
      FF ff(detsets, 34);
      ff.resize(4);
      std::copy(sv.begin(), sv.begin() + 4, ff.begin());
    }
    DSTV::Range r = detsets.equal_range(30, cmp10());
    REQUIRE(r.second - r.first == 2);
    r = detsets.equal_range(40, cmp10());
    REQUIRE(r.second - r.first == 0);

    SECTION("find") {
      REQUIRE(detsets.exists(22));
      REQUIRE(!detsets.exists(44));
      DST df = *detsets.find(22);
      REQUIRE(df.id() == 22);
      REQUIRE(df.size() == 2);
    }
    SECTION("indexing") {
      DST df = detsets[22];
      REQUIRE(df.id() == 22);
      REQUIRE(df.size() == 2);
    }

    SECTION("find not found") {
      DSTV::const_iterator p = detsets.find(44);
      REQUIRE(p == detsets.end());
    }
    SECTION("invalid index") {
      REQUIRE_THROWS_MATCHES(
          detsets[44], edm::Exception, Catch::Matchers::Predicate<edm::Exception>([](edm::Exception const &e) {
            return e.categoryCode() == edm::errors::InvalidReference;
          }));
    }
  }

  SECTION("algorithm") {
    DSTV detsets(2);
    for (unsigned int n = 1; n < 5; ++n) {
      unsigned int id = 20 + n;
      FF ff(detsets, id);
      ff.resize(n);
      std::copy(sv.begin(), sv.begin() + n, ff.begin());
    }
    {
      FF ff(detsets, 31);
      ff.resize(2);
      std::copy(sv.begin(), sv.begin() + 2, ff.begin());
    }
    {
      FF ff(detsets, 11);
      ff.resize(2);
      std::copy(sv.begin(), sv.begin() + 2, ff.begin());
    }
    {
      FF ff(detsets, 34);
      ff.resize(4);
      std::copy(sv.begin(), sv.begin() + 4, ff.begin());
    }
    DSTV::Range r = detsetRangeFromPair(detsets, acc(3));
    REQUIRE(r.second - r.first == 2);
    r = edmNew::detsetRangeFromPair(detsets, acc(4));
    REQUIRE(r.second - r.first == 0);
    std::vector<DSTV::data_type const *> v;
    edmNew::copyDetSetRange(detsets, v, acc(3));
    VerifyAlgos va(v);
    edmNew::foreachDetSetObject(detsets, acc(3), va);
  }

  SECTION("onDemand") {
    auto pg = std::make_shared<Getter>(&sv);
    Getter &g = *pg;
    REQUIRE(!g.aborted);
    std::vector<unsigned int> v = {21, 23, 25, 27, 1020};
    DSTV detsets(pg, v, 2);
    REQUIRE(g.ntot == 0);
    REQUIRE(detsets.onDemand());
    REQUIRE(detsets.exists(21));
    REQUIRE(!detsets.exists(22));
    REQUIRE(!detsets.isValid(21));
    REQUIRE(!detsets.isValid(22));
    detsets.reserve(5, 100);
    DST df = *detsets.find(21, true);
    REQUIRE(df.id() == 21);
    REQUIRE(df.size() == 1);
    REQUIRE(detsets.isValid(21));
    REQUIRE(!detsets.isValid(23));
    REQUIRE(g.ntot == 1);
    REQUIRE(!g.aborted);
    {
      DST df = detsets[25];
      REQUIRE(df.id() == 25);
      REQUIRE(df.size() == 5);
      REQUIRE(g.ntot == 1 + 5);
      REQUIRE(!g.aborted);
    }
    {
      DST df = detsets[1020];
      REQUIRE(df.id() == 1020);
      REQUIRE(df.size() == 0);
      REQUIRE(g.ntot == 1 + 5);
      REQUIRE(g.aborted);
    }
    // no onDemand!
    int i = 0;
    for (auto di = detsets.begin(); di != detsets.end(); ++di) {
      ++i;
      auto ds = *di;
      auto id = ds.id();
      REQUIRE((id == 1020 || (id > 20 && id < 28 && id % 2 == 1)));
      if (1020 == id || 21 == id || 25 == id)
        REQUIRE(ds.isValid());
      else
        REQUIRE(!ds.isValid());
    }
    REQUIRE(5 == i);
    REQUIRE(g.ntot == 1 + 5);
    REQUIRE(std::for_each(detsets.begin(true), detsets.end(true), VerifyIter(&sv, 1, 2)).n == 9);
    REQUIRE(std::for_each(detsets.begin(true), detsets.end(true), VerifyIter(&sv, 1, 2)).n == 9);
    REQUIRE(g.ntot == 1 + 3 + 5 + 7);
    SECTION("find not found") {
      DSTV::const_iterator p = detsets.find(22);
      REQUIRE(p == detsets.end());
    }

    SECTION("invalid index") {
      REQUIRE_THROWS_MATCHES(
          detsets[22], edm::Exception, Catch::Matchers::Predicate<edm::Exception>([](edm::Exception const &e) {
            return e.categoryCode() == edm::errors::InvalidReference;
          }));
    }
    DSTV detsets2;
    detsets2.swap(detsets);
    REQUIRE(detsets2.onDemand());
    DSTV detsets3;
    detsets3 = std::move(detsets2);
    REQUIRE(detsets3.onDemand());
    DSTV detsets5(std::move(detsets3));
    REQUIRE(detsets5.onDemand());
    DSTV detsets4(detsets5);
    REQUIRE(detsets4.onDemand());
  }

  SECTION("toRangeMap") {
    DSTV detsets(2);
    for (unsigned int n = 1; n < 5; ++n) {
      unsigned int id = 20 + n;
      FF ff(detsets, id);
      ff.resize(n);
      std::copy(sv.begin(), sv.begin() + n, ff.begin());
    }
    {
      FF ff(detsets, 31);
      ff.resize(2);
      std::copy(sv.begin(), sv.begin() + 2, ff.begin());
    }
    {
      FF ff(detsets, 11);
      ff.resize(2);
      std::copy(sv.begin(), sv.begin() + 2, ff.begin());
    }
    {
      FF ff(detsets, 34);
      ff.resize(4);
      std::copy(sv.begin(), sv.begin() + 4, ff.begin());
    }
    SECTION("edmNew::copy") {
      typedef edm::RangeMap<det_id_type, edm::OwnVector<B> > RM;
      edm::RangeMap<det_id_type, edm::OwnVector<B> > rm;
      edmNew::copy(detsets, rm);
      rm.post_insert();
      std::vector<det_id_type> ids = rm.ids();
      REQUIRE(ids.size() == detsets.size());
      REQUIRE(rm.size() == detsets.dataSize());
      for (int i = 0; i < int(ids.size()); i++) {
        RM::range r = rm.get(ids[i]);
        DST df = *detsets.find(ids[i]);
        REQUIRE(static_cast<unsigned long>(r.second - r.first) == df.size());
        REQUIRE(std::equal(r.first, r.second, df.begin()));
      }
    }
  }
}
