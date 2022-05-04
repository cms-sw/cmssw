#pragma once

#include<atomic>
#include<array>
#include<memory>
#include<algorithm>
#include<cassert>
#include<mutex>
#include <vector>
#include <cstdint>
#include<iostream>
#include<chrono>

namespace poolDetails {

 constexpr int bucket(uint64_t s) { return 64-__builtin_clzl(s-1); }
 constexpr uint64_t bucketSize(int b) { return 1LL<<b;}

};


class SimplePoolAllocator {

public:

  using Pointer = void *;


  virtual ~SimplePoolAllocator() = default;

  virtual Pointer doAlloc(size_t size) =0;
  virtual void doFree(Pointer ptr) = 0;

  SimplePoolAllocator(int maxSlots) : m_maxSlots(maxSlots) {
    for ( auto & p : m_used) p = true;
  }
  
  int size() const { return m_size;}

  Pointer pointer(int i) const { return m_slots[i]; }

  void free(int i) {
    m_last[i] = -1;
    m_used[i]=false;
  }

  int alloc(uint64_t s) {
    auto i = allocImpl(s);

    //test garbage
    // if(totBytes>4507964512) garbageCollect();

    if (i>=0) {
       assert(m_used[i]);
       if (nullptr==m_slots[i]) std::cout << "race ??? " << i << ' ' << m_bucket[i] << ' ' << m_last[i] << std::endl;
       assert(m_slots[i]);
       return i;
    } 
    garbageCollect();
    i =  allocImpl(s);
    if (i>=0) { assert(m_used[i]); assert(m_slots[i]);assert(m_last[i]>=0);}
    return i;
  }

  int allocImpl(uint64_t s) {
    auto b = poolDetails::bucket(s);
    assert(s<=poolDetails::bucketSize(b));
    int ls = size();
    // look for an existing slot
    for (int i=0; i<ls; ++i) {
      if (b!=m_bucket[i]) continue;    
      if (m_used[i]) continue;
      bool exp = false;
      if (m_used[i].compare_exchange_strong(exp,true)) {
        // verify if in the mean time the garbage collector did operate
        if(nullptr == m_slots[i]) {
          assert(m_bucket[i]<0);
          m_used[i] = false;
          continue;
        }
        m_last[i] = 0;
        return i;
      }
    }

    // try to create in existing slot (if garbage has been collected)
    ls = useOld(b);
    if (ls>=0) return ls;

    // try to allocate a new slot
    if (m_size>=m_maxSlots) return -1;
    ls = m_size++;
    if (ls>=m_maxSlots) return -1;
    m_last[ls] = 2;
    return createAt(ls,b);
  }

  int createAt(int ls, int b) {
    assert(m_used[ls]);
    assert(m_last[ls]>0);
    m_bucket[ls]=b;
    auto as = poolDetails::bucketSize(b);
    assert(nullptr==m_slots[ls]);
    m_slots[ls] = doAlloc(as);
    if (nullptr == m_slots[ls]) return -1;
    totBytes+=as;
    nAlloc++;
    return ls;
  }

  void garbageCollect() {
    int ls = size();
    for (int i=0; i<ls; ++i) {
      if (m_used[i]) continue;
      if (m_bucket[i]<0) continue; 
      bool exp = false;
      if (!m_used[i].compare_exchange_strong(exp,true)) continue;
      assert(m_used[i]);
      if( nullptr != m_slots[i]) {
        assert(m_bucket[i]>=0);  
        doFree(m_slots[i]);
        nFree++;
        totBytes-= poolDetails::bucketSize(m_bucket[i]);
      }
      m_slots[i] = nullptr;
      m_bucket[i] = -1;
      m_last[i] = -3;
      m_used[i] = false; // here memory fence as well
    }
  }


  int useOld(int b) {
    int ls = size();
    for (int i=0; i<ls; ++i) {
      if ( m_bucket[i]>=0) continue;
      if (m_used[i]) continue;
      bool exp = false;
      if (!m_used[i].compare_exchange_strong(exp,true)) continue;
      if( nullptr != m_slots[i]) { // ops allocated and freed
        assert(m_bucket[i]>=0);
        assert(m_last[i] = -1);
        m_used[i] = false;
        continue;
      }
      assert(m_used[i]);
      m_last[i] = 1;
      return createAt(i,b);
    }
    return -1;
  }

  void dumpStat() const {
   uint64_t fn=0; 
   uint64_t fs=0;
   int ls = size();
   for (int i=0; i<ls; ++i) {
      if (m_used[i]) {
        auto b = m_bucket[i];
        if (b<0) continue;
        fn++;
        fs += (1LL<<b);
      }
   }
   std::cout << "# slots " << size() << '\n'
              << "# bytes " << totBytes << '\n'
              << "# alloc " << nAlloc << '\n'
              << "# free " << nFree << '\n'
              << "# used " << fn << ' ' << fs << '\n'
              << std::endl;
  }
 

private:

  const int m_maxSlots;

  std::vector<int> m_last = std::vector<int>(m_maxSlots,-2);


  std::vector<int> m_bucket = std::vector<int>(m_maxSlots,-1);
  std::vector<Pointer> m_slots = std::vector<Pointer>(m_maxSlots,nullptr);
  std::vector<std::atomic<bool>> m_used = std::vector<std::atomic<bool>>(m_maxSlots);
  std::atomic<int> m_size=0;

  std::atomic<uint64_t> totBytes = 0;
  std::atomic<uint64_t> nAlloc = 0;
  std::atomic<uint64_t> nFree = 0;

};


template<typename Traits>
struct SimplePoolAllocatorImpl final : public SimplePoolAllocator {

  SimplePoolAllocatorImpl(int maxSlots) : SimplePoolAllocator(maxSlots){}

  ~SimplePoolAllocatorImpl() override = default;

  Pointer doAlloc(size_t size) override { return Traits::alloc(size);}
  void doFree(Pointer ptr) override { Traits::free(ptr);}

};


#include <cstdlib>
struct PosixAlloc {

  using Pointer = void *;

  static Pointer alloc(size_t size) { return ::malloc(size); }
  static void free(Pointer ptr) { ::free(ptr); }

};


