#include "FWCore/Utilities/interface/clone_ptr.h"

#include<iostream>
#include<algorithm>
#include<cassert>

int cla=0;

struct A{
   A(){}
   A(int j) : i(j){}

  A * clone() const { cla++; /*std::cout<< "c A " << i << std::endl;*/ return new A(*this);}

  int i=3;
};


int da=0;
int da0=0;
struct B{

  B(){}
  B(B const &b) : a(b.a){}
  B(B &&b)   noexcept : a(std::move(b.a)){}

  B & operator=(B const &b) {
    a=b.a;
    return *this;
  }
  B & operator=(B&&b)  noexcept {
    a=std::move(b.a);
    return *this;
  }

  ~B() {if(a) da++; else da0++; /*std::cout<< "d B " << (a ? a->i : -99) << std::endl;*/}

  extstd::clone_ptr<A> a;
};


#include<vector>
int main() {

  B b; b.a.reset(new A(2));

  assert(cla==0);
  B c = b;
  assert(cla==1);
  B d = b;
  assert(cla==2);

  b.a.reset(new A(-2));

  //std::cout<< c.a->i << std::endl;
  assert(c.a->i == 2);
  assert(b.a->i == -2);
  
  c = b;
  assert(cla==3);

  assert(c.a->i == -2);
  //std::cout<< c.a->i << std::endl;
  c.a.reset(new A(-7));
  assert(c.a->i == -7);
  assert(cla==3);
  
  //std::cout << cla << " " << da << " " << da0 << std::endl;

  std::vector<B> vb(1); 
  vb.push_back(b);
  assert(cla==4);

  int expectedCla = cla;
  int cValue = c.a->i;
  vb.push_back(std::move(c));
  //some compilers will still cause a clone to be called and others won't
  //assert(cla==expectedCla);
  assert((*vb.back().a).i ==cValue);
  expectedCla = cla+1;
  vb[0]=d;

  //std::cout <<cla <<" "<<expectedCla<<std::endl;
  assert(cla==expectedCla);
  assert(vb[0].a->i == d.a->i);
  // assert(cla==5);
  // assert(da==0);
  //std::cout << cla << " " << da << " " << da0 << std::endl;

  //std::cout<< vb[0].a->i << std::endl;
  assert(vb[0].a->i == 2);
  std::sort(vb.begin(),vb.end(),[](B const & rh, B const & lh){return rh.a->i<lh.a->i;});
  assert( (*vb[0].a).i == -7);
  assert(cla==expectedCla);
  //std::cout<< (*vb[0].a).i << std::endl;
  std::swap(b,d);
  assert(cla==expectedCla);
  // assert(da==0);
  //std::cout << cla << " " << da << " " << da0 << std::endl;


  return 0;
}
