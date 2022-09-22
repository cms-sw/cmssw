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
#include <cmath>

#define MEMORY_POOL_DEBUG

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

  struct TriState {
    static constexpr int free = 0;
    static constexpr int used = 1;
    static constexpr int scheduled = -1;
    alignas(64) std::atomic<int> v;
  };

class SimplePoolAllocator {
public:
  using Pointer = void *;

  static constexpr void * invalidStream = nullptr; //(void*)(3UL);  // "3" cannot be a pointer

  virtual ~SimplePoolAllocator() = default;

  virtual Pointer doAlloc(size_t size) = 0;
  virtual void doFree(Pointer ptr) = 0;
  virtual void scheduleFree(memoryPool::Payload *payload, void *stream) = 0;

  SimplePoolAllocator(int maxSlots) : m_maxSlots(maxSlots) {
    for (auto &p : m_used)
      p.v = TriState::used;
  }

  int size() const { return m_size; }

  Pointer pointer(int i) const { return m_slots[i]; }

  void dumpStat() const;

  void free(int i, bool sched=true) {
    if (sched) assert(m_used[i].v == TriState::scheduled);
#ifdef MEMORY_POOL_DEBUG
    m_last[i] = -1;
#endif
    m_used[i].v = TriState::free;
    m_stream[i] = invalidStream;
  }

  int alloc(uint64_t s, void * stream) {
    auto i = allocImpl(s,stream);

    //test garbage
    // if(totBytes>4507964512) garbageCollect();

    if (i >= 0) {
#ifdef MEMORY_POOL_DEBUG
      assert(m_used[i].v==TriState::used);
      if (nullptr == m_slots[i])
        std::cout << "race ??? " << i << ' ' << m_bucket[i] << ' ' << m_last[i] << std::endl;
      assert(m_slots[i]);
#endif
      return i;
    }
    garbageCollect();
    i = allocImpl(s,stream);
    if (i >= 0) {
      assert(m_used[i].v==TriState::used);
      assert(m_slots[i]);
#ifdef MEMORY_POOL_DEBUG
      assert(m_last[i] >= 0);
#endif
    }
    return i;
  }

protected:
  int allocImpl(uint64_t s, void * stream);
  int createAt(int ls, int b, void * stream);
  void garbageCollect();
  int useOld(int b, void * stream);


  const int m_maxSlots;

#ifdef MEMORY_POOL_DEBUG
  std::vector<int> m_last = std::vector<int>(m_maxSlots, -2);
#endif

  std::vector<int> m_bucket = std::vector<int>(m_maxSlots, -1);
  std::vector<Pointer> m_slots = std::vector<Pointer>(m_maxSlots, nullptr);
  std::vector<Pointer> m_stream = std::vector<Pointer>(m_maxSlots, invalidStream);
  std::vector<TriState> m_used = std::vector<TriState>(m_maxSlots);
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

  using SimplePoolAllocator::SimplePoolAllocator;

  ~SimplePoolAllocatorImpl() override {
    garbageCollect();
#ifdef MEMORY_POOL_DEBUG
    dumpStat();
#endif
  }

  Pointer doAlloc(size_t size) override { return Traits::alloc(size); }
  void doFree(Pointer ptr) override { Traits::free(ptr); }

  void scheduleFree(memoryPool::Payload *payload, void *stream) override {
    assert(payload->pool == this);
    auto const &buckets = payload->buckets;
    for (auto i : buckets) {
      m_used[i].v = TriState::scheduled;
    }
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
