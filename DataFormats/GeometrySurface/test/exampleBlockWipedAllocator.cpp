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
	  , public  SizeBlockWipedAllocated<A1> 
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
	  , public  SizeBlockWipedAllocated<A2> 
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
	  , public  SizeBlockWipedAllocated<A3> 
#endif
{
  virtual A3* clone() const {
    return new A3(*this);
  }
  
  float a;
  char c[12];
  long long  l;

};

#ifdef __GXX_EXPERIMENTAL_CXX0X__
struct E0
{
  virtual ~E0(){}
  virtual E0 * clone() const =0;
  virtual unsigned int size() const =0;
  typedef BWAFactory<E0> Factory;
  typedef Factory::UP UP;
  virtual UP cloneTransient() const=0;
protected:
  template<typename T>
  static UP cloneMe(T const & t) {
    return Factory::create<T>(t);
  }

  int i;

};


struct E1 : public E0
{
  virtual E1 * clone() const { 
    return new E1(*this);
  }
  
  virtual E0::UP cloneTransient() const {
    return cloneMe(*this);
  }
  virtual unsigned int size() const { return sizeof(*this); }

  float a;
  char c[12];
  long long  l;

};

struct A4 : public B
#ifdef BALLOC
	  , public  SizeBlockWipedAllocated<A3> 
#endif
{

  explicit A4(E0 * it) : me(it){}
  explicit A4(E0 const & it) : me(it.cloneTransient()){}
  
  void releaseIt() {
    me.release();
  }
  void cloneIt() {
    E0::UP it = me.release()->cloneTransient();
    me = std::move(it);
  }

  virtual A4* clone() const {
    return nullptr;
  }
  
  float a;
  char c[12];
  long long  l;

  E0::UP me;

};


#endif


struct D : public BlockWipedAllocated<D> {
 
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
 
bool noPool=true;

void dump(std::string const & mess="") {
#ifdef DEBUG
  std::cout << mess << std::endl;
#ifdef PBALLOC
  std::cout << "BlockAllocator stat"<< std::endl;
  std::cout << "still alive " << BlockWipedPoolAllocated::s_alive << std::endl;
  if (noPool) return;
  Dumper dumper;
  blockWipedPool().visit(dumper);
#endif
#ifdef BALLOC
  BlockWipedAllocator::Stat sa1 = SizeBlockWipedAllocated<A1>::stat();
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
typedef boost::shared_ptr<D> DP;

bool flop=false;

void gen(BP & bp) {
  static bool flip=false;
  if (flip) 
    bp.reset(new A1);
  else
    bp.reset(flop ? (B*)(new A2): (B*)(new A3));
  flip = !flip;
}



void go() {

  bool started=true;
  LocalCache<A4> lc0;
  dump("start");

  for (int i=0;i<500;i++) {
    {
      flop = !flop;
      if (!noPool) {
	blockWipedPool().wipe();
	if (i%10==0)  blockWipedPool().clear();
      }
      BP b1(new A1);
      BP b2(new A2);
      BP b3(new A3);
      DP d(new D);
      dump();
      {
	BP bb1(b1->clone());
	BP bb2(b2->clone());
	BP bb3(b3->clone());
	DP d2(new D(*d));
	dump();
      }
      dump("after clone destr");
      
      BP b11(new A1);
      BP b22(new A2);
      BP b23(new A2);
      DP d2(new D);
      dump();
      b1.reset();
      b2.reset();
      b3.reset();
      d.reset();
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
      std::vector<E1> es(2002);
      for (int i=0;i<2002;i++){
	static LocalCache<A4> * lc;
	// to allow multiple loop: fake static init
	if (started) { 
	  started=false;
	  lc =  &lc0;
	}
	std::auto_ptr<A4> & ptr = lc->ptr;
	if (ptr.get()) {
	  ptr->~A4();
	  new(ptr.get()) A4(&es[i]);
	} else {
	  ptr.reset(new A4(&es[i]));
	}
	if (i%5==0) {
	  ptr->cloneIt();
	  vev.push_back(BP(ptr.release()));
	}else ptr->releaseIt();  
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
  if (!noPool) blockWipedPool().clear();
  dump("the end");

  std::cout << "\n" << sizeof(B)
	    << " " << sizeof(A1)
	    << " " << sizeof(A2)
	    << " " << sizeof(A3)
	    << " " << sizeof(D)
	    << " " << sizeof(E1)
	    << std::endl;

  
  std::cout << "\nEND OF JOB" << std::endl;

}
int  main() {
  noPool=true;
  {
    std::cout << "no pool" << std::endl;
    go();
  }
  noPool=false; 
  {
    BlockWipedPoolAllocated::s_alive=0;
    std::cout << "unused pool" << std::endl;
    BlockWipedPool pool(1096,1096);
    blockWipedPool(&pool);
    go();
  }
  
  {
    std::cout << "used pool" << std::endl;
    BlockWipedPoolAllocated::s_alive=0;

    BlockWipedPool pool(1096,1096);
    blockWipedPool(&pool);
    BlockWipedPoolAllocated::usePool();
    go();
  }
  return 0;
}

