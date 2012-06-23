#ifndef BlockWipedAllocator_H
#define BlockWipedAllocator_H

#include<vector>
#include<list>
// #include<map>
#include <ext/hash_map>

#include <algorithm>
#include<memory>

#include<boost/bind.hpp>

#include "FWCore/Utilities/interface/Visibility.h"


// #include<iostream>

/*  Allocator that never removes single allocations
 *  it "wipes" or "clears" the whole allocation when not needed anymore
 *  if not wiped it may easily run out of memory
 */
class BlockWipedAllocator {
public:
  BlockWipedAllocator( std::size_t typeSize,
		       std::size_t blockSize,
		       std::size_t  maxRecycle
		       );
  

  /*  copy constructor clone the allocator and the memory it manages
   *  it needs to be reinitialized to avoid pointing to "rh"
   *
   */
  BlockWipedAllocator(BlockWipedAllocator const & rh);

  BlockWipedAllocator& operator=(BlockWipedAllocator const & rh);
  ~BlockWipedAllocator();

  void * alloc();
 
  void dealloc(void *);

  // redime memory to the system heap
  void clear() const;

  // reset allocator status. does not redime memory
  void wipe(bool force=true) const;


  // longliving (static) local caches: to be reset at wipe
  struct LocalCache {
    virtual ~LocalCache(){}
    virtual void reset()=0;
  };

  void registerCache(LocalCache * c) {
    localCaches.push_back(c);
  }

private:
  std::vector<LocalCache*> localCaches;
  std::vector<void *> recycled;


protected:

  BlockWipedAllocator & me() const;

public:

  struct Stat {
    size_t typeSize;
    size_t blockSize;
    size_t currentOccupancy;
    size_t currentAvailable;
    std::ptrdiff_t totalAvailable;
    size_t nBlocks;
    int alive;
  };
  
  Stat stat() const;
  
private:
  void nextBlock(bool advance) dso_internal;


  struct Block {
    std::size_t m_allocated;
    std::vector<unsigned char> m_data;
  };

  typedef unsigned char * pointer; 
  typedef std::list<Block> Blocks;
  typedef Blocks::iterator iterator;
  typedef Blocks::const_iterator const_iterator;


  std::size_t m_typeSize;
  std::size_t m_blockSize;
  std::size_t m_maxRecycle;
  pointer m_next;
  iterator m_current;
  Blocks m_blocks;

  int m_alive; // for stat purposes

};


class BlockWipedPool {
public:
  typedef BlockWipedAllocator Allocator;
  //  typedef std::map<std::size_t, Allocator> Pool;
  typedef __gnu_cxx::hash_map<std::size_t, Allocator> Pool;

  BlockWipedPool(std::size_t blockSize, std::size_t  maxRecycle);
  ~BlockWipedPool();

  Allocator & allocator( std::size_t typeSize);

  void wipe(bool force=true);

  void clear();

  template<typename Visitor>
  void visit(Visitor& visitor) const {
    std::for_each(m_pool.begin(),m_pool.end(),boost::bind(&Visitor::visit,visitor,
							  boost::bind(&Pool::value_type::second,_1)
							  ));
  }


private:
  std::size_t m_blockSize;
  std::size_t m_maxRecycle;
  Pool m_pool;
  Allocator * m_last;
  std::size_t m_lastSize;
};



/*  generaric Basic class
 * 
 */
class BlockWipedPoolAllocated {
public:
  virtual ~BlockWipedPoolAllocated(){}
  // instance counter...
  static int s_alive;
  static void * operator new(size_t s, void * p);
  static void * operator new(size_t s);
  
  static void operator delete(void * p, size_t s);
  
  static BlockWipedAllocator & allocator(size_t s);
  

  static BlockWipedAllocator::Stat stat(size_t s);
  
  // throw id s_alive!=0???
  static void usePool();


  // private:
  static bool s_usePool;
  // static BlockAllocator * s_allocator;
};

// singleton
BlockWipedPool & blockWipedPool(BlockWipedPool * p=0);

template<size_t S>
BlockWipedAllocator & blockWipedAllocator() {
  static BlockWipedAllocator & local = blockWipedPool().allocator(S);
  return local;
}

template<typename T>
struct LocalCache : public BlockWipedAllocator::LocalCache {
  std::auto_ptr<T> ptr;
  LocalCache(){ 
    if (BlockWipedPoolAllocated::s_usePool)
      blockWipedAllocator<sizeof(T)>().registerCache(this);
  }
  ~LocalCache(){}
  void reset(){ ptr.reset();}
};


/*  Allocator by type
 * 
 */
template<typename T>
class BlockWipedAllocated {
public:
  static void * operator new(size_t) {
    return alloc();
  }
  
  static void operator delete(void * p) {
    dealloc(p);
  }
  
  static void * operator new(size_t, void * p) {
    return p;
  }

  static void * alloc() {
    BlockWipedPoolAllocated::s_alive++;
    return (BlockWipedPoolAllocated::s_usePool) ? allocator().alloc()  : ::operator new(sizeof(T));
  }
  
  static void dealloc(void * p) {
    if (0==p) return;
    BlockWipedPoolAllocated::s_alive--;
    return (BlockWipedPoolAllocated::s_usePool) ? allocator().dealloc(p)  : ::operator delete(p);
  }
  
#ifdef __GXX_EXPERIMENTAL_CXX0X__
  template<typename... Args>
  static void
  construct(T *  p, Args&&... args)
      { ::new(p) T(std::forward<Args>(args)...); }
#endif


  static void destroy( T * p) {  p->~T(); }


  static BlockWipedAllocator & allocator() {
    static BlockWipedAllocator & local = blockWipedPool().allocator(sizeof(T));
    return local;
  }
  

  static BlockWipedAllocator::Stat stat() {
    return allocator().stat();
  }
  

private:
  
  // static BlockAllocator * s_allocator;
};

#ifdef __GXX_EXPERIMENTAL_CXX0X__
template<typename B>
struct BWADestroy {
    BWADestroy() {}
    void operator () (B* p) { 
      if (0==p) return;
      BlockWipedPoolAllocated::s_alive--;
      if (BlockWipedPoolAllocated::s_usePool) {
	BlockWipedAllocator & local =  blockWipedPool().allocator(p->size());
	p->~B();
	local.dealloc(p);
      }
      else { delete p; }
    }
  };

template<typename B>
struct BWAFactory {
  typedef  BWADestroy<B> Destroy;
  typedef std::unique_ptr<B,  BWAFactory<B>::Destroy> UP;
  template<typename T, typename... Args>
  static UP create(Args&&... args) {
    // create derived, destroy base
    UP ret( (T*)BlockWipedAllocated<T>::alloc(),
	    Destroy()
	    );
    BlockWipedAllocated<T>::construct((T*)ret.get(),std::forward<Args>(args)...);
    return ret;
  }
};
#endif



/*  Allocator by size (w/o pool)
 * 
 */
template<typename T>
class SizeBlockWipedAllocated {
public:
  static void * operator new(size_t) {
    return allocator().alloc();
  }
  
  static void operator delete(void * p) {
    allocator().dealloc(p);
  }
  
  static BlockWipedAllocator & allocator() {
    static BlockWipedAllocator & local = blockWipedAllocator<sizeof(T)>();
    return  local;
  }
  

  static BlockWipedAllocator::Stat stat() {
    return allocator().stat();
  }
  
private:
  
  // static BlockAllocator * s_allocator;
};


#endif // BlockAllocator_H
