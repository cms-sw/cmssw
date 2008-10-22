#include "DataFormats/GeometrySurface/interface/BlockWipedAllocator.h"


BlockWipedAllocator::BlockWipedAllocator( std::size_t typeSize,
					  std::size_t blockSize):
  m_typeSize(typeSize), m_blockSize(blockSize), alive(0){
  wipe();
}
  

BlockWipedAllocator::BlockWipedAllocator(BlockWipedAllocator const & rh) :
  m_typeSize(rh.m_typeSize), m_blockSize(rh.m_blockSize),alive(0) {
  wipe();
}

BlockWipedAllocator& BlockWipedAllocator::operator=(BlockWipedAllocator const & rh) {
  m_typeSize=rh.m_typeSize; m_blockSize=rh.m_blockSize;
  alive=0;
  wipe();
  return *this;
}
    

void * BlockWipedAllocator::alloc() {
  alive++;
  void * ret = m_next;
  m_next+=m_typeSize;
  Block & block = *m_current;
  ++block.m_allocated;
  if(m_next==(&block.m_data.back())+1)
    nextBlock(true);
  return ret;
}
  
void BlockWipedAllocator::dealloc(void *) {
  alive--;
}

void BlockWipedAllocator::clear() const {
  me().m_blocks.clear();
  me().wipe();
}

void BlockWipedAllocator::wipe() const {
  me().m_current=me().m_blocks.begin();
  me().nextBlock(false);
}
  
BlockWipedAllocator & BlockWipedAllocator::me() const {
  return const_cast<BlockWipedAllocator&>(*this);
}

BlockWipedAllocator::Stat BlockWipedAllocator::stat() const {
  Stat s = { m_typeSize, m_blockSize, (*m_current).m_allocated,
	     (&*(*m_current).m_data.end()-m_next)/m_typeSize,
	     std::distance(const_iterator(m_current),m_blocks.end()),
	     m_blocks.size()};
  return s;
}

void BlockWipedAllocator::nextBlock(bool advance) {
  if (advance) m_current++;
  if (m_current==m_blocks.end()) {
    m_blocks.push_back(Block());
    m_current=m_blocks.end(); --m_current;
  }
  m_current->m_data.resize(m_blockSize*m_typeSize);
  m_current->m_allocated=0;
  m_next = &(m_current->m_data.front());
}


BlockWipedPool::BlockWipedPool(std::size_t blockSize) : m_blockSize(blockSize){}


BlockWipedPool::Allocator & BlockWipedPool::allocator( std::size_t typeSize) {
  Pool::iterator p=m_pool.find(typeSize);
  if (p!=m_pool.end()) return (*p).second;
  return (*m_pool.insert(std::make_pair(typeSize,Allocator(typeSize, m_blockSize))).first).second;
}

void BlockWipedPool::wipe() {
  std::for_each(m_pool.begin(),m_pool.end(),boost::bind(&Allocator::wipe,
							boost::bind(&Pool::value_type::second,_1)
							));
}

void BlockWipedPool::clear() {
  std::for_each(m_pool.begin(),m_pool.end(),boost::bind(&Allocator::clear,
							boost::bind(&Pool::value_type::second,_1)
							));
}




BlockWipedPool & blockWipedPool() {
  static BlockWipedPool local(1024);
  return local;
}

void * BlockWipedPoolAllocated::operator new(size_t s) {
  return allocator(s).alloc();
}

void BlockWipedPoolAllocated::operator delete(void * p) {
  allocator().dealloc(p);
}

BlockWipedAllocator & BlockWipedPoolAllocated::allocator(size_t s) {
  return  blockWipedPool().allocator(s);
}


BlockWipedAllocator::Stat BlockWipedPoolAllocated::stat(size_t s) {
  return allocator(s).stat();
}
  
