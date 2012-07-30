#include "DataFormats/GeometrySurface/interface/BlockWipedAllocator.h"
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <algorithm>

#define PBALLOC
#define DEBUG

struct B 
#ifdef PBALLOC
  : public BlockWipedPoolAllocated
#endif
{
  virtual ~B(){}
  virtual B* clone() const =0;
  int i;

};




struct A1 : public B
#ifdef BALLOC
	  , public BlockWipedAllocated<A1> 
#endif
{
  virtual A1* clone() const {
    return new A1(*this);
  }

  double a;
  char c;
  bool b;

};

struct A2 : public B
#ifdef BALLOC
	  , public BlockWipedAllocated<A2> 
#endif
{
  virtual A2* clone() const {
    return new A2(*this);
  }
  
  float a;
  char c[12];
  long long  l;

};

struct A3 : public B
#ifdef BALLOC
	  , public BlockWipedAllocated<A3> 
#endif
{
  virtual A3* clone() const {
    return new A3(*this);
  }
  
  float a;
  char c[12];
  long long  l;

};

/*
BlockAllocated<A1>::s_allocator =  BlockAllocated<A1>::allocator(10);
BlockAllocated<A2>::s_allocator =  BlockAllocated<A2>::allocator(100);
*/

struct Dumper {
  void visit(BlockWipedAllocator const& alloc) const {
    BlockWipedAllocator::Stat sa1 = alloc.stat();
    std::cout << "Alloc for size " << sa1.typeSize
	      << ": " << sa1.blockSize
	      << ", " << sa1.currentOccupancy
	      << "/" << sa1.currentAvailable
	      << ", " << sa1.totalAvailable
	      << "/" << sa1.nBlocks
      	      << ", " << sa1.alive
	      << std::endl;
  }
  
};
 
void dump(std::string const & mess="") {
#ifdef DEBUG
  std::cout << mess << std::endl;
#ifdef PBALLOC
  std::cout << "BlockAllocator stat"<< std::endl;
  std::cout << "still alive " << BlockWipedPoolAllocated::s_alive << std::endl;
  Dumper dumper;
  blockWipedPool().visit(dumper);
#endif
#ifdef BALLOC
  BlockWipedAllocator::Stat sa1 = BlockWipedAllocated<A1>::stat();
  BlockWipedAllocator::Stat sa2 = SizeBlockWipedAllocated<A2>::stat();
  std::cout << "A1 " << sa1.blockSize
	    << " " << sa1.currentOccupancy
	    << " " << sa1.currentAvailable
	    << " " << sa1.totalAvailable
	    << " " << sa1.nBlocks       	  
	    << ", " << sa1.alive
	    << std::endl;
  std::cout << "A2 " << sa2.blockSize
	    << " " << sa2.currentOccupancy
	    << " " << sa2.currentAvailable
	    << " " << sa2.totalAvailable
	    << " " << sa2.nBlocks
	    << ", " << sa2.alive
 	    << std::endl;
#endif
#endif
}

typedef boost::shared_ptr<B> BP;

bool flop=false;

void gen(BP & bp) {
  static bool flip=false;
  if (flip) 
    bp.reset(new A1);
  else
    bp.reset(flop ? (B*)(new A2): (B*)(new A3));
  flip = !flip;
}

int main(int v, char **) {
  BlockWipedPool pool(1096,1096);
  blockWipedPool(&pool);
  if (v==1) BlockWipedPoolAllocated::usePool();

  dump();

  for (int i=0;i<500;i++) {
    {
      flop = !flop;
      blockWipedPool().wipe();
      if (i%10==0)  blockWipedPool().clear();
      BP b1(new A1);
      BP b2(new A2);
      BP b3(new A3);
      dump();
      {
	BP bb1(b1->clone());
	BP bb2(b2->clone());
	BP bb3(b3->clone());
	dump();
      }
      dump("after clone destr");
      
      BP b11(new A1);
      BP b22(new A2);
      BP b23(new A2);
      dump();
      b1.reset();
      b2.reset();
      b3.reset();
      dump("after first destr");
    }
    dump();
    { 
      std::vector<BP> v(233);
      std::for_each(v.begin(),v.end(),&gen);
      dump("after 233 alloc");
      v.resize(123);
      dump("after 110 distr");
    }
    dump();
    
    {
      std::vector<BP> vev;
      for (int i=0;i<2002;i++){
	static LocalCache<A3> lc;
	std::auto_ptr<A3> & ptr = lc.ptr;
	if (ptr.get()) {
	  ptr->~A3();
	  new(ptr.get()) A3;
	} else {
	  ptr.reset(new A3);
	}
	if (i%5==0) vev.push_back(BP(ptr.release()));			       
      }
      dump("after 2002/5 local caches");
    
      for (int i=0;i<3;i++){
	std::vector<BP> v(2432);
	std::for_each(v.begin(),v.end(),&gen);
	std::vector<BP> v1(3213);
	std::for_each(v1.begin(),v1.end(),&gen);
	{
	  std::vector<BP> d; d.swap(v);
	}
	// alloc disalloc
	std::vector<BP> vs(514);
	std::for_each(vs.begin(),vs.end(),&gen);
	std::for_each(vs.begin(),vs.end(),&gen);
      }
    }
    dump("loop end");
  }
  std::cout << sizeof(B)
	    << " " << sizeof(A1)
	    << " " << sizeof(A2)
	    << " " << sizeof(A3)
	    << std::endl;
  return 0;
}

