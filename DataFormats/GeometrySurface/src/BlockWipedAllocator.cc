#include "DataFormats/GeometrySurface/interface/BlockWipedAllocator.h"
#include "FWCore/Utilities/interface/Likely.h"

BlockWipedAllocator::BlockWipedAllocator( std::size_t typeSize,
					  std::size_t blockSize,
					  std::size_t  maxRecycle):
  m_typeSize(typeSize), m_blockSize(blockSize), m_maxRecycle(maxRecycle), m_alive(0){
  //  if (typeSize<32) abort(); // throw std::bad_alloc();
  recycled.reserve(m_maxRecycle);
  wipe();
}
  

BlockWipedAllocator::BlockWipedAllocator(BlockWipedAllocator const & rh) :
  m_typeSize(rh.m_typeSize), m_blockSize(rh.m_blockSize), m_maxRecycle(rh.m_maxRecycle), m_alive(0) {
  recycled.reserve(m_maxRecycle);
  wipe();
}

BlockWipedAllocator& BlockWipedAllocator::operator=(BlockWipedAllocator const & rh) {
  m_typeSize=rh.m_typeSize; m_blockSize=rh.m_blockSize; m_maxRecycle=rh.m_maxRecycle;
  recycled.reserve(m_maxRecycle);
  m_alive=0;
  wipe();
  return *this;
}
 

BlockWipedAllocator::~BlockWipedAllocator() {
  clear();
}

   
// cannot keep the count as dealloc is never called...

void * BlockWipedAllocator::alloc() {
  m_alive++;
  if likely(!recycled.empty()) {
    void * ret = recycled.back();
    recycled.pop_back();
    return ret;
  } 
  void * ret = m_next;
  m_next+=m_typeSize;
  Block & block = *m_current;
  ++block.m_allocated;
  if unlikely(m_next==(&block.m_data.back())+1) nextBlock(true);
  return ret;
}
  
void BlockWipedAllocator::dealloc(void * p) {
  if likely (recycled.size()<m_maxRecycle) recycled.push_back(p);
  m_alive--;
}

void BlockWipedAllocator::clear() const {
  me().m_blocks.clear();
  me().wipe();
}

void BlockWipedAllocator::wipe(bool force) const {
  if (m_alive>0 && !force) return;
  // reset caches
  std::for_each(localCaches.begin(),localCaches.end(),boost::bind(&LocalCache::reset,_1));

  me().m_current=me().m_blocks.begin();
  me().nextBlock(false);
  me().recycled.clear();
}
  
BlockWipedAllocator & BlockWipedAllocator::me() const {
  return const_cast<BlockWipedAllocator&>(*this);
}

BlockWipedAllocator::Stat BlockWipedAllocator::stat() const {
  Stat s = { m_typeSize, m_blockSize, (*m_current).m_allocated,
	     (&*(*m_current).m_data.end()-m_next)/m_typeSize,
	     std::distance(const_iterator(m_current),m_blocks.end()),
	     m_blocks.size(), m_alive};
  return s;
}

void BlockWipedAllocator::nextBlock(bool advance) {
  if likely(advance) m_current++;
  if unlikely(m_current==m_blocks.end()) {
    m_blocks.push_back(Block());
    m_current=m_blocks.end(); --m_current;
  }
  m_current->m_data.resize(m_blockSize*m_typeSize);
  m_current->m_allocated=0;
  m_next = &(m_current->m_data.front());
}


BlockWipedPool::BlockWipedPool(std::size_t blockSize, std::size_t  maxRecycle) : 
  m_blockSize(blockSize),  m_maxRecycle(maxRecycle), m_last(0), m_lastSize(0){}

BlockWipedPool::~BlockWipedPool() {clear();}



BlockWipedPool::Allocator & BlockWipedPool::allocator( std::size_t typeSize) {
  if likely(m_lastSize==typeSize) return *m_last;
  Pool::iterator p=m_pool.find(typeSize);
  m_lastSize=typeSize;
  if likely (p!=m_pool.end())  return *(m_last = &(*p).second);
  return *(m_last=&(*m_pool.insert(std::make_pair(typeSize,Allocator(typeSize, m_blockSize, m_maxRecycle))).first).second);
}

void BlockWipedPool::wipe(bool force) {
  std::for_each(m_pool.begin(),m_pool.end(),boost::bind(&Allocator::wipe,
							boost::bind(&Pool::value_type::second,_1),force
							));
}

void BlockWipedPool::clear() {
  std::for_each(m_pool.begin(),m_pool.end(),boost::bind(&Allocator::clear,
							boost::bind(&Pool::value_type::second,_1)
							));
}




BlockWipedPool & blockWipedPool(BlockWipedPool * p) {
  static BlockWipedPool * local=0;
  if (p!=0) local=p;
  return *local;
}


int BlockWipedPoolAllocated::s_alive=0;
bool BlockWipedPoolAllocated::s_usePool=false;

void BlockWipedPoolAllocated::usePool() { 
  // throw id s_alive!=0???
  if (0==s_alive) s_usePool=true;
}



void * BlockWipedPoolAllocated::operator new(size_t s) {
  s_alive++;
  return (s_usePool) ? allocator(s).alloc() : ::operator new(s);
}

void *  BlockWipedPoolAllocated::operator new(size_t s, void * p) {
  return p;
}

#include<typeinfo>
#include<iostream>
struct AQ {
  virtual ~AQ(){}
};
void BlockWipedPoolAllocated::operator delete(void * p, size_t s) {
  if (0==p) return;
  // if (s<100) std::cout << typeid(*(BlockWipedPoolAllocated*)(p)).name() << std::endl;
  s_alive--;
  (s_usePool) ? allocator(s).dealloc(p) : ::operator delete(p);

}

BlockWipedAllocator & BlockWipedPoolAllocated::allocator(size_t s) {
  return  blockWipedPool().allocator(s);
}


BlockWipedAllocator::Stat BlockWipedPoolAllocated::stat(size_t s) {
  return allocator(s).stat();
}
  

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
int ReferenceCountedPoolAllocated::s_alive=0;
int ReferenceCountedPoolAllocated::s_referenced=0;
