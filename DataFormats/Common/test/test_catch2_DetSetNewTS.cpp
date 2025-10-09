#include "catch2/catch_all.hpp"

#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetAlgorithm.h"
#include "DataFormats/Common/interface/DetSet2RangeMap.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <vector>
#include <algorithm>
#include <mutex>
#include <iostream>
#include <atomic>
#include <thread>
#include <memory>

typedef std::mutex Mutex;
typedef std::unique_lock<std::mutex> Lock;

class TestDetSet {
public:
  template <typename T>
  static auto& data(edmNew::DetSetVector<T>& detsets) {
    return detsets.m_data;
  }
};

namespace {
  template <typename T>
  inline void spinlock(std::atomic<T> const& lock, T val) {
    while (lock.load(std::memory_order_acquire) != val) {
    }
  }

  template <typename T>
  inline void spinlockSleep(std::atomic<T> const& lock, T val) {
    while (lock.load(std::memory_order_acquire) != val) {
      nanosleep(0, 0);
    }
  }

  // syncronize all threads in a parallel section (for testing purposes)
  void sync(std::atomic<int>& all, int total) {
    ++all;
    spinlock(all, total);
  }

  unsigned int number_of_threads() {
    auto nThreads = std::thread::hardware_concurrency();
    return nThreads == 0 ? 1 : nThreads;
  }
  template <typename F>
  void parallel_run(F iFunc) {
    std::vector<std::thread> threads;
    auto nThreads = number_of_threads();
    for (unsigned int i = 0; i < nThreads; ++i) {
      threads.emplace_back([i, nThreads, iFunc] { iFunc(i, nThreads); });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  struct B {
    virtual ~B() {}
    virtual B* clone() const = 0;
  };

  struct T : public B {
    T(int iv = 0) : v(iv) {}
    int v;
    bool operator==(T t) const { return v == t.v; }
    virtual T* clone() const { return new T(*this); }
  };

  typedef edmNew::DetSetVector<T> DSTV;
  typedef edmNew::DetSet<T> DST;
  typedef edmNew::det_id_type det_id_type;
  typedef DSTV::FastFiller FF;
  typedef DSTV::TSFastFiller TSFF;

  void read(DSTV const& detsets, bool all = false) {
    for (auto di = detsets.begin(false); di != detsets.end(false); ++di) {
      auto ds = *di;
      auto id = ds.id();
      //std::cout << id << ' ';
      if (ds.isValid()) {
        REQUIRE(ds[0] == 100 * (id - 20) + 3);
        REQUIRE(ds[1] == -(100 * (id - 20) + 3));
      }
    }
    //std::cout << std::endl;
  }

  struct Getter final : public DSTV::Getter {
    Getter(int& nth) : ntot(0), nth_(nth) {}
    void fill(TSFF& ff) const override {
      int n = ff.id() - 20;
      REQUIRE(n >= 0);
      REQUIRE(ff.size() == 0);
      ff.push_back((100 * n + 3));
      REQUIRE(ff.size() == 1);
      REQUIRE(ff[0] == 100 * n + 3);
      ff.push_back(-(100 * n + 3));
      REQUIRE(ff.size() == 2);
      REQUIRE(ff[1] == -(100 * n + 3));
      ntot.fetch_add(1, std::memory_order_acq_rel);
    }
    mutable std::atomic<unsigned int> ntot;
    int& nth_;
  };
}  // namespace
TEST_CASE("DetSetNewTS", "[DetSetNewTS]") {
  int nth = number_of_threads();
  std::vector<DSTV::data_type> sv(10);
  DSTV::data_type v[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::copy(v, v + 10, sv.begin());

  SECTION("infrastructure") {
    //std::cout << std::endl;
    for (int i = 0; i < 10; i++) {
      int a = 0;
      std::atomic<int> b(0);
      std::atomic<int> lock(0);
      std::atomic<int> nt(number_of_threads());
      parallel_run([&a, &b, &lock, &nt](unsigned int, unsigned int) {
        sync(lock, nt);
        a++;
        b.fetch_add(1, std::memory_order_acq_rel);
      });
      //if (i == 5)
      //  std::cout << "threads " << lock << " " << a << ' ' << b << std::endl;
      REQUIRE(b == nt);
      a = 0;
      b = 0;
      parallel_run([&a, &b](unsigned int, unsigned int) {
        a++;
        b.fetch_add(1, std::memory_order_acq_rel);
      });
      //if (i == 5)
      //  std::cout << "threads " << lock << " " << a << ' ' << b << std::endl;
      nth = nt;
    }
  }

  SECTION("fillSeq") {
    //std::cout << std::endl;
    DSTV detsets(2);
    std::atomic<int> lock(0);
    std::atomic<int> idet(0);
    std::atomic<int> trial(0);
    int maxDet = 100 * nth;
    parallel_run([&lock, &idet, &trial, &detsets, maxDet](unsigned int, unsigned int numberOfThreads) {
      sync(lock, numberOfThreads);
      while (true) {
        int ldet = idet;
        if (!(ldet < maxDet))
          break;
        while (!idet.compare_exchange_weak(ldet, ldet + 1))
          ;
        if (ldet >= maxDet)
          break;
        unsigned int id = 20 + ldet;
        bool done = false;
        while (!done) {
          try {
            {
              FF ff(detsets, id);
              ff.push_back(100 * ldet + 3);
              REQUIRE(TestDetSet::data(detsets).back().v == (100 * ldet + 3));
              ff.push_back(-(100 * ldet + 3));
              REQUIRE(TestDetSet::data(detsets).back().v == -(100 * ldet + 3));
            }
            done = true;
          } catch (edm::Exception const&) {
            trial++;
          }
        }
      }
    });
    //std::cout << idet << ' ' << detsets.size() << std::endl;
    read(detsets, true);
    REQUIRE(int(detsets.size()) == maxDet);
    //std::cout << "trials " << trial << std::endl;
  }

  SECTION("fillPar") {
    //std::cout << std::endl;
    auto pg = std::make_shared<Getter>(nth);
    Getter& g = *pg;
    int maxDet = 100 * nth;
    std::vector<unsigned int> v(maxDet);
    int k = 20;
    for (auto& i : v)
      i = k++;
    DSTV detsets(pg, v, 2);
    detsets.reserve(maxDet, 100 * maxDet);
    REQUIRE(g.ntot == 0);
    REQUIRE(detsets.onDemand());
    REQUIRE(maxDet == int(detsets.size()));
    std::atomic<int> lock(0);
    std::atomic<int> idet(0);
    //std::atomic<int> count(0);
    DST df31 = detsets[31];
    //std::cout << "start parallel section" << std::endl;
    parallel_run([&lock, &detsets, &idet, maxDet, &g](unsigned int threadNumber, unsigned int numberOfThreads) {
      sync(lock, numberOfThreads);
      if (threadNumber % 2 == 0) {
        DST df = detsets[25];
        REQUIRE(df.id() == 25);
        REQUIRE(df.size() == 2);
        REQUIRE(df[0] == 100 * (25 - 20) + 3);
        REQUIRE(df[1] == -(100 * (25 - 20) + 3));
      }
      while (true) {
        if (threadNumber == 0)
          read(detsets);
        int ldet = idet.load(std::memory_order_acquire);
        if (!(ldet < maxDet))
          break;
        while (!idet.compare_exchange_weak(ldet, ldet + 1, std::memory_order_acq_rel))
          ;
        if (ldet >= maxDet)
          break;
        unsigned int id = 20 + ldet;
        {
          DST df = *detsets.find(id, true);
          REQUIRE(int(g.ntot) > 0);
          REQUIRE(df.id() == id);
          REQUIRE(df.size() == 2);
          REQUIRE(df[0] == 100 * (id - 20) + 3);
          REQUIRE(df[1] == -(100 * (id - 20) + 3));
        }
        if (threadNumber == 1)
          read(detsets);
      }
    });
    //std::cout << "end parallel section" << std::endl;
    REQUIRE(df31.id() == 31);
    REQUIRE(df31.size() == 2);
    REQUIRE(df31[0] == 100 * (31 - 20) + 3);
    REQUIRE(df31[1] == -(100 * (31 - 20) + 3));
    //std::cout << "summary " << idet << ' ' << detsets.size() << ' ' << g.ntot << ' ' << count << std::endl;
    read(detsets, true);
    REQUIRE(int(g.ntot) == maxDet);
    REQUIRE(int(detsets.size()) == maxDet);
  }
}
