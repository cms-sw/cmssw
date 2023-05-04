#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetAlgorithm.h"
#include "DataFormats/Common/interface/DetSet2RangeMap.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <vector>
#include <algorithm>

#include <mutex>
typedef std::mutex Mutex;
// typedef std::lock_guard<std::mutex> Lock;
typedef std::unique_lock<std::mutex> Lock;

namespace global {
  // control cout....
  Mutex coutLock;
}  // namespace global

#include <iostream>
#include <atomic>
#include <thread>

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

namespace {
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
}  // namespace

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

bool operator==(T const& t, B const& b) {
  T const* p = dynamic_cast<T const*>(&b);
  return p && p->v == t.v;
}

bool operator==(B const& b, T const& t) { return t == b; }

typedef edmNew::DetSetVector<T> DSTV;
typedef edmNew::DetSet<T> DST;
typedef edmNew::det_id_type det_id_type;
typedef DSTV::FastFiller FF;
typedef DSTV::TSFastFiller TSFF;

class TestDetSet : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestDetSet);
  CPPUNIT_TEST(infrastructure);
  CPPUNIT_TEST(fillSeq);
  CPPUNIT_TEST(fillPar);

  CPPUNIT_TEST_SUITE_END();

public:
  TestDetSet();
  ~TestDetSet() {}
  void setUp() {}
  void tearDown() {}

  void infrastructure();
  void fillSeq();
  void fillPar();

public:
  int nth = 1;
  std::vector<DSTV::data_type> sv;
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDetSet);

TestDetSet::TestDetSet() : sv(10) {
  DSTV::data_type v[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::copy(v, v + 10, sv.begin());
  nth = number_of_threads();
}

void read(DSTV const& detsets, bool all = false) {
  for (auto di = detsets.begin(false); di != detsets.end(false); ++di) {
    auto ds = *di;
    auto id = ds.id();
    std::cout << id << ' ';
    if (ds.isValid()) {
      CPPUNIT_ASSERT(ds[0] == 100 * (id - 20) + 3);
      CPPUNIT_ASSERT(ds[1] == -(100 * (id - 20) + 3));
    }
  }
  std::cout << std::endl;
}

void TestDetSet::infrastructure() {
  std::cout << std::endl;
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

    if (i == 5)
      std::cout << "threads " << lock << " " << a << ' ' << b << std::endl;
    CPPUNIT_ASSERT(b == nt);
    a = 0;
    b = 0;

    parallel_run([&a, &b](unsigned int, unsigned int) {
      a++;
      b.fetch_add(1, std::memory_order_acq_rel);
    });
    if (i == 5)
      std::cout << "threads " << lock << " " << a << ' ' << b << std::endl;

    nth = nt;
  }
}

void TestDetSet::fillSeq() {
  std::cout << std::endl;

  DSTV detsets(2);
  // unsigned int ntot=0;

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
            FF ff(detsets, id);  // serialize
            ff.push_back(100 * ldet + 3);
            CPPUNIT_ASSERT(detsets.m_data.back().v == (100 * ldet + 3));
            ff.push_back(-(100 * ldet + 3));
            CPPUNIT_ASSERT(detsets.m_data.back().v == -(100 * ldet + 3));
          }
          // read(detsets);  // cannot read in parallel while filling in this case
          done = true;
        } catch (edm::Exception const&) {
          trial++;
          //read(detsets);
        }
      }
    }
    // read(detsets);
  });

  std::cout << idet << ' ' << detsets.size() << std::endl;
  read(detsets, true);
  CPPUNIT_ASSERT(int(detsets.size()) == maxDet);
  std::cout << "trials " << trial << std::endl;
}

struct Getter final : public DSTV::Getter {
  Getter(TestDetSet* itest) : ntot(0), test(*itest) {}

  void fill(TSFF& ff) override {
    int n = ff.id() - 20;
    CPPUNIT_ASSERT(n >= 0);
    CPPUNIT_ASSERT(ff.size() == 0);
    ff.push_back((100 * n + 3));
    CPPUNIT_ASSERT(ff.size() == 1);
    CPPUNIT_ASSERT(ff[0] == 100 * n + 3);
    ff.push_back(-(100 * n + 3));
    CPPUNIT_ASSERT(ff.size() == 2);
    CPPUNIT_ASSERT(ff[1] == -(100 * n + 3));
    ntot.fetch_add(1, std::memory_order_acq_rel);
  }

  std::atomic<unsigned int> ntot;
  TestDetSet& test;
};

void TestDetSet::fillPar() {
  std::cout << std::endl;
  auto pg = std::make_shared<Getter>(this);
  Getter& g = *pg;
  int maxDet = 100 * nth;
  std::vector<unsigned int> v(maxDet);
  int k = 20;
  for (auto& i : v)
    i = k++;
  DSTV detsets(pg, v, 2);
  detsets.reserve(maxDet, 100 * maxDet);
  CPPUNIT_ASSERT(g.ntot == 0);
  CPPUNIT_ASSERT(detsets.onDemand());
  CPPUNIT_ASSERT(maxDet == int(detsets.size()));

  std::atomic<int> lock(0);
  std::atomic<int> idet(0);

  std::atomic<int> count(0);

  DST df31 = detsets[31];

  std::cout << "start parallel section" << std::endl;
  parallel_run([&lock, &detsets, &idet, maxDet, &g](unsigned int threadNumber, unsigned int numberOfThreads) {
    sync(lock, numberOfThreads);
    if (threadNumber % 2 == 0) {
      DST df = detsets[25];  // everybody!
      CPPUNIT_ASSERT(df.id() == 25);
      CPPUNIT_ASSERT(df.size() == 2);
      CPPUNIT_ASSERT(df[0] == 100 * (25 - 20) + 3);
      CPPUNIT_ASSERT(df[1] == -(100 * (25 - 20) + 3));
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
        CPPUNIT_ASSERT(int(g.ntot) > 0);
        assert(df.id() == id);
        assert(df.size() == 2);
        assert(df[0] == 100 * (id - 20) + 3);
        assert(df[1] == -(100 * (id - 20) + 3));
      }
      if (threadNumber == 1)
        read(detsets);
    }
  });
  std::cout << "end parallel section" << std::endl;

  CPPUNIT_ASSERT(df31.id() == 31);
  CPPUNIT_ASSERT(df31.size() == 2);
  CPPUNIT_ASSERT(df31[0] == 100 * (31 - 20) + 3);
  CPPUNIT_ASSERT(df31[1] == -(100 * (31 - 20) + 3));

  std::cout << "summary " << idet << ' ' << detsets.size() << ' ' << g.ntot << ' ' << count << std::endl;
  read(detsets, true);
  CPPUNIT_ASSERT(int(g.ntot) == maxDet);
  CPPUNIT_ASSERT(int(detsets.size()) == maxDet);
}
