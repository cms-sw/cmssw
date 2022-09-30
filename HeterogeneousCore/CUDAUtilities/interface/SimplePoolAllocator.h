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

#define MEMORY_POOL_DEBUG

namespace poolDetails {

  constexpr int bucket(uint64_t s) { return 64 - __builtin_clzl(s - 1); }
  constexpr uint64_t bucketSize(int b) { return 1LL << b; }

};  // namespace poolDetails

class SimplePoolAllocator;

namespace memoryPool {
  struct Payload {
    SimplePoolAllocator *pool;
    std::vector<std::pair<int, uint64_t>> buckets;
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

  static constexpr void *invalidStream = nullptr;

  virtual ~SimplePoolAllocator() = default;

  virtual Pointer doAlloc(size_t size) = 0;
  virtual void doFree(Pointer ptr) = 0;
  virtual void scheduleFree(memoryPool::Payload *payload, void *stream) = 0;

  SimplePoolAllocator(int maxSlots, bool useScheduled) : m_maxSlots(maxSlots), m_useScheduled(useScheduled) {
    for (auto &p : m_used)
      p.v = TriState::used;
  }

  int size() const { return m_size; }

  Pointer pointer(int i) const { return m_slots[i]; }
  uint64_t count(int i) const { return m_count[i]; }
  void setScheduled(int i) { m_used[i].v = TriState::scheduled; }

  void dumpStat() const;

  void free(int i, uint64_t count) {
    if (m_used[i].v != TriState::scheduled)
      return;
    if (count != m_count[i])
      return;
    int exp = TriState::scheduled;
    if (m_used[i].v.compare_exchange_strong(exp, TriState::used)) {  // used! so nobody can touch it...
      if (count != m_count[i]) {                                     // oops
        m_used[i].v = TriState::scheduled;
        return;
      }
#ifdef MEMORY_POOL_DEBUG
      m_last[i] = -1;
#endif
      m_stream[i] = invalidStream;
      assert(count == m_count[i]);  // should we reset to 0?
      m_used[i].v = TriState::free;
    }
  }

  int alloc(uint64_t s, void *stream) {
    auto i = allocImpl(s, stream);

    //test garbage
    // if(totBytes>4507964512) garbageCollect();

    if (i >= 0) {
#ifdef MEMORY_POOL_DEBUG
      assert(m_used[i].v == TriState::used);
      if (nullptr == m_slots[i])
        std::cout << "race ??? " << i << ' ' << m_bucket[i] << ' ' << m_last[i] << std::endl;
      assert(m_slots[i]);
#endif
      return i;
    }
    garbageCollect();
    i = allocImpl(s, stream);
    if (i >= 0) {
      assert(m_used[i].v == TriState::used);
      assert(m_slots[i]);
#ifdef MEMORY_POOL_DEBUG
      assert(m_last[i] >= 0);
#endif
    }
    return i;
  }

  // slow. needed by standard allocators
  int index(Pointer p) const {
    int ls = size();
    for (int i = 0; i < ls; ++i)
      if (m_slots[i] == p)
        return i;
    return -1;
  }

protected:
  int allocImpl(uint64_t s, void *stream);
  int createAt(int ls, int b, void *stream);
  void garbageCollect();
  int useOld(int b, void *stream);

  const int m_maxSlots;
  const bool m_useScheduled;

#ifdef MEMORY_POOL_DEBUG
  std::vector<int> m_last = std::vector<int>(m_maxSlots, -2);
#endif

  std::vector<int> m_bucket = std::vector<int>(m_maxSlots, -1);
  std::vector<Pointer> m_slots = std::vector<Pointer>(m_maxSlots, nullptr);
  std::vector<Pointer> m_stream = std::vector<Pointer>(m_maxSlots, invalidStream);
  std::vector<TriState> m_used = std::vector<TriState>(m_maxSlots);
  std::vector<uint64_t> m_count = std::vector<uint64_t>(m_maxSlots, 0);
  std::atomic<int> m_size = 0;

  uint64_t maxBytes = 0;
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
      pool.free(i.first, i.second);
    }
    delete payload;
  }
}  // namespace poolDetails

template <typename T>
struct SimplePoolAllocatorImpl final : public SimplePoolAllocator {
  using Traits = T;

  SimplePoolAllocatorImpl(int maxSlots) : SimplePoolAllocator(maxSlots, Traits::useScheduled) {}

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
      m_used[i.first].v = TriState::scheduled;
    }
    Traits::scheduleFree(payload, stream);
  }
};

#include <cstdlib>
struct PosixAlloc {
  using Pointer = void *;

  static constexpr bool useScheduled = false;

  static Pointer alloc(size_t size) { return ::malloc(size); }
  static void free(Pointer ptr) { ::free(ptr); }

  static void scheduleFree(memoryPool::Payload *payload, void *) { poolDetails::freeAsync(payload); }
};
