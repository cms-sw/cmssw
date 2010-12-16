#ifndef BlockWipedAllocator_H
#define BlockWipedAllocator_H

#include<vector>
#include<list>
#include<map>
#include <algorithm>

#include<boost/bind.hpp>

// #include<iostream>

/*  Allocator that never removes single allocations
 *  it "wipes" or "clears" the whole allocation when not needed anymore
 *  if not wiped it may easily run out of memory
 */
class BlockWipedAllocator {
public:
  BlockWipedAllocator( std::size_t typeSize,
		       std::size_t blockSize);
  

  /*  copy constructor clone the allocator and the memory it manages
   *  it needs to be reinitialized to avoid pointing to "rh"
   *
   */
  BlockWipedAllocator(BlockWipedAllocator const & rh);

  BlockWipedAllocator& operator=(BlockWipedAllocator const & rh);
    

  void * alloc();
 
  void dealloc(void *);

  // redime memory to the system heap
  void clear() const;

  // reset allocator status. does not redime memory
  void wipe() const;


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
  

protected:

  BlockWipedAllocator & me() const;

public:

  struct Stat {
    size_t typeSize;
    size_t blockSize;
    size_t currentOccupancy;
    size_t currentAvailable;
    size_t totalAvailable;
    size_t nBlocks;
    int alive;
  };
  
  Stat stat() const;
  
private:
  void nextBlock(bool advance);


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
  pointer m_next;
  iterator m_current;
  Blocks m_blocks;

  int m_alive; // for stat purposes

};


class BlockWipedPool {
public:
  typedef BlockWipedAllocator Allocator;
  typedef std::map<std::size_t, Allocator> Pool; 

  BlockWipedPool(std::size_t blockSize);

  Allocator & allocator( std::size_t typeSize);

  void wipe();

  void clear();

  template<typename Visitor>
  void visit(Visitor& visitor) const {
    std::for_each(m_pool.begin(),m_pool.end(),boost::bind(&Visitor::visit,visitor,
							  boost::bind(&Pool::value_type::second,_1)
							  ));
  }


private:
  std::size_t m_blockSize;
  Pool m_pool;
};


// singleton
BlockWipedPool & blockWipedPool();

template<size_t S>
BlockWipedAllocator & blockWipedAllocator() {
  static BlockWipedAllocator & local = blockWipedPool().allocator(S);
  return local;
}

template<typename T>
struct LocalCache : public BlockWipedAllocator::LocalCache {
  std::auto_ptr<T> ptr;
  LocalCache(){ 
    blockWipedAllocator<sizeof(T)>().registerCache(this);
  }
  ~LocalCache(){}
  void reset(){ ptr.reset();}
};


/*  generaric Basic class
 * 
 */
class BlockWipedPoolAllocated {
public:
  // instance counter...
  static int s_alive;
  static void * operator new(size_t s, void * p);
  static void * operator new(size_t s);
  
  static void operator delete(void * p, size_t s);
  
  static BlockWipedAllocator & allocator(size_t s);
  

  static BlockWipedAllocator::Stat stat(size_t s);
  
  // throw id s_alive!=0???
  static void usePool();


private:
  static bool s_usePool;
  // static BlockAllocator * s_allocator;
};

// below: not used

/*  Allocator by type
 * 
 */
template<typename T>
class BlockWipedAllocated {
public:
  static void * operator new(size_t) {
    return allocator().alloc();
  }
  
  static void operator delete(void * p) {
    allocator().dealloc(p);
  }
  
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


/*  Allocator by size
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
