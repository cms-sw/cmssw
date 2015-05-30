#include "CommonTools/Utils/interface/DynArray.h"



struct A {

int i=-3;
double k=0.1;

virtual ~A(){}

};


#include<cassert>
#include<iostream>

int main(int s, char **) {

  using T=A;

  unsigned n = 4*s;

//  alignas(alignof(T)) unsigned char a_storage[sizeof(T)*n];
//  DynArray<T> a(a_storage,n);

  declareDynArray(T,n,a);

  // T b[n];
  declareDynArray(T,n,b);


  auto pa = [&](auto i) { a[1].k=0.3; return a[i].k; };
  auto pb = [&](auto i) { b[1].k=0.5; return b[i].k; };

  auto loop = [&](auto k) { for(auto const & q : a) k = std::min(k,q.k); return k;};

  std::cout << a[n-1].k << ' ' << pa(1) << ' ' << loop(2.3) << std::endl;
  std::cout << b[n-1].k << ' ' << pb(1) << std::endl;

  initDynArray(bool,n,q,true);
  if (q[n-1]) std::cout << "ok" << std::endl;	

  return 0;
};
