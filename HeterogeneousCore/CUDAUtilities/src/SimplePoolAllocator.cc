#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"

int SimplePoolAllocator::allocImpl(uint64_t s, void * stream) {
  auto b = poolDetails::bucket(s);
  assert(s <= poolDetails::bucketSize(b));
  int ls = size();
  // look for an existing slot
  for (int i = 0; i < ls; ++i) {
    if (b != m_bucket[i])
      continue;
    if (m_used[i].v!=TriState::free)
      continue;
    int exp = TriState::free;
    if (m_used[i].v.compare_exchange_strong(exp, TriState::used)) {
      // verify if in the mean time the garbage collector did operate
      if (nullptr == m_slots[i]) {
        assert(m_bucket[i] < 0);
        m_used[i].v = TriState::free;
        continue;
      }
#ifdef MEMORY_POOL_DEBUG
      m_last[i] = 0;
#endif
      m_stream[ls] = stream;
      return i;
    }
  }

  // try to create in existing slot (if garbage has been collected)
  ls = useOld(b,stream);
  if (ls >= 0)
    return ls;

  // try to allocate a new slot
  if (m_size >= m_maxSlots)
    return -1;
  ls = m_size++;
  if (ls >= m_maxSlots)
    return -1;
#ifdef MEMORY_POOL_DEBUG
  m_last[ls] = 2;
#endif
  return createAt(ls, b, stream);
}

int SimplePoolAllocator::createAt(int ls, int b, void * stream) {
  assert(m_used[ls].v==TriState::used);
#ifdef MEMORY_POOL_DEBUG
  assert(m_last[ls] > 0);
#endif
  m_bucket[ls] = b;
  auto as = poolDetails::bucketSize(b);
  assert(nullptr == m_slots[ls]);
  m_slots[ls] = doAlloc(as);
  if (nullptr == m_slots[ls]) {
    m_bucket[ls] = -1;
    m_used[ls].v = false;
    return -1;
  }
  m_stream[ls] = stream;
  totBytes += as;
  nAlloc++;
  return ls;
}

void SimplePoolAllocator::garbageCollect() {
  int ls = size();
  for (int i = 0; i < ls; ++i) {
    if (m_used[i].v!=TriState::free)
      continue;
    if (m_bucket[i] < 0)
      continue;
    int exp = TriState::free;
    if (!m_used[i].v.compare_exchange_strong(exp,TriState::used))
      continue;
    assert(m_used[i].v==TriState::used);
    if (nullptr != m_slots[i]) {
      assert(m_bucket[i] >= 0);
      doFree(m_slots[i]);
      nFree++;
      totBytes -= poolDetails::bucketSize(m_bucket[i]);
    }
    m_slots[i] = nullptr;
    m_bucket[i] = -1;
#ifdef MEMORY_POOL_DEBUG
    m_last[i] = -3;
#endif
    m_stream[i] = invalidStream;
    m_used[i].v = false;  // here memory fence as well
  }
}

int SimplePoolAllocator::useOld(int b, void * stream) {
  int ls = size();
  for (int i = 0; i < ls; ++i) {
    if (m_bucket[i] >= 0)
      continue;
    if (m_used[i].v!=TriState::free)
      continue;
    int exp =TriState::free;
    if (!m_used[i].v.compare_exchange_strong(exp,TriState::used))
      continue;
    if (nullptr != m_slots[i]) {  // ops allocated and freed
      assert(m_bucket[i] >= 0);
#ifdef MEMORY_POOL_DEBUG
      assert(m_last[i] = -1);
#endif
      m_used[i].v = TriState::free;
      continue;
    }
    assert(m_used[i].v==TriState::used);
#ifdef MEMORY_POOL_DEBUG
    m_last[i] = 1;
#endif
    return createAt(i, b, stream);
  }
  return -1;
}

void SimplePoolAllocator::dumpStat() const {
  uint64_t fn = 0;
  uint64_t fs = 0;
  int ls = size();
  for (int i = 0; i < ls; ++i) {
    if (m_used[i].v==TriState::used) {
      auto b = m_bucket[i];
      if (b < 0)
        continue;
      fn++;
      fs += (1LL << b);
    }
  }
  std::cout << "# slots " << size() << '\n'
            << "# bytes " << totBytes << '\n'
            << "# alloc " << nAlloc << '\n'
            << "# free " << nFree << '\n'
            << "# used " << fn << ' ' << fs << '\n'
            << std::endl;
}
