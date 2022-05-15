#pragma once

#include <atomic>
#include <array>
#include <memory>
#include <algorithm>
#include <cassert>
#include <mutex>
#include <vector>
#include <cstdint>
#include <iostream>
#include <chrono>

// #define MEMORY_POOL_DEBUG

namespace poolDetails {

  constexpr int bucket(uint64_t s) { return 64 - __builtin_clzl(s - 1); }
  constexpr uint64_t bucketSize(int b) { return 1LL << b; }

};  // namespace poolDetails

class SimplePoolAllocator;

namespace memoryPool {
  struct Payload {
    SimplePoolAllocator *pool;
    std::vector<int> buckets;
  };
}  // namespace memoryPool

class SimplePoolAllocator {
public:
  using Pointer = void *;

  virtual ~SimplePoolAllocator() = default;

  virtual Pointer doAlloc(size_t size) = 0;
  virtual void doFree(Pointer ptr) = 0;
  virtual void scheduleFree(memoryPool::Payload *payload, void *stream) = 0;

  SimplePoolAllocator(int maxSlots) : m_maxSlots(maxSlots) {
    for (auto &p : m_used)
      p.v = true;
  }

  int size() const { return m_size; }

  Pointer pointer(int i) const { return m_slots[i]; }

  void dumpStat() const;

  void free(int i) {
#ifdef MEMORY_POOL_DEBUG
    m_last[i] = -1;
#endif
    m_used[i].v = false;
  }

  int alloc(uint64_t s) {
    auto i = allocImpl(s);

    //test garbage
    // if(totBytes>4507964512) garbageCollect();

    if (i >= 0) {
#ifdef MEMORY_POOL_DEBUG
      assert(m_used[i].v);
      if (nullptr == m_slots[i])
        std::cout << "race ??? " << i << ' ' << m_bucket[i] << ' ' << m_last[i] << std::endl;
      assert(m_slots[i]);
#endif
      return i;
    }
    garbageCollect();
    i = allocImpl(s);
    if (i >= 0) {
      assert(m_used[i].v);
      assert(m_slots[i]);
#ifdef MEMORY_POOL_DEBUG
      assert(m_last[i] >= 0);
#endif
    }
    return i;
  }

protected:
  int allocImpl(uint64_t s);
  int createAt(int ls, int b);
  void garbageCollect();
  int useOld(int b);

private:
  const int m_maxSlots;

#ifdef MEMORY_POOL_DEBUG
  std::vector<int> m_last = std::vector<int>(m_maxSlots, -2);
#endif

  std::vector<int> m_bucket = std::vector<int>(m_maxSlots, -1);
  std::vector<Pointer> m_slots = std::vector<Pointer>(m_maxSlots, nullptr);
  struct alBool {
    alignas(64) std::atomic<bool> v;
  };
  std::vector<alBool> m_used = std::vector<alBool>(m_maxSlots);
  std::atomic<int> m_size = 0;

  std::atomic<uint64_t> totBytes = 0;
  std::atomic<uint64_t> nAlloc = 0;
  std::atomic<uint64_t> nFree = 0;
};

namespace poolDetails {
  //  free callback
  inline void freeAsync(memoryPool::Payload *payload) {
    auto &pool = *(payload->pool);
    auto const &buckets = payload->buckets;
    for (auto i : buckets) {
      pool.free(i);
    }
    delete payload;
  }
}  // namespace poolDetails

template <typename T>
struct SimplePoolAllocatorImpl final : public SimplePoolAllocator {
  using Traits = T;

  SimplePoolAllocatorImpl(int maxSlots) : SimplePoolAllocator(maxSlots) {}

  ~SimplePoolAllocatorImpl() override {
    garbageCollect();
    //#ifdef MEMORY_POOL_DEBUG
    dumpStat();
    //#endif
  }

  Pointer doAlloc(size_t size) override { return Traits::alloc(size); }
  void doFree(Pointer ptr) override { Traits::free(ptr); }

  void scheduleFree(memoryPool::Payload *payload, void *stream) override {
    assert(payload->pool == this);
    Traits::scheduleFree(payload, stream);
  }
};

#include <cstdlib>
struct PosixAlloc {
  using Pointer = void *;

  static Pointer alloc(size_t size) { return ::malloc(size); }
  static void free(Pointer ptr) { ::free(ptr); }

  static void scheduleFree(memoryPool::Payload *payload, void *) { poolDetails::freeAsync(payload); }
};
