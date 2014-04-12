#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"

#include<vector>
#include<iostream>
#include<algorithm>

int main() {

  using PD=mayown_ptr<double>;
  using VD=std::vector<PD>;

  double d1=2;
  PD p1; assert(p1.empty()); assert(!p1.isOwn()); p1.reset();
  p1.reset(d1); assert(&d1 == &(*p1));

  std::vector<double> dd(10,3.14);
  VD vd;
  for (int i=0; i<10; ++i) {
    if (i%2==0) vd.push_back(PD(dd[i]));
    else vd.push_back(PD(new double(-1.2)));
  }

  VD vd2; std::swap(vd,vd2);  assert(10==vd2.size());
  for (auto i=0U; i<vd2.size(); ++i) std::cout << *vd2[i] << ' '; std::cout << std::endl;
  for (auto i=0U; i<vd2.size(); ++i) if (i%2==0) assert(!vd2[i].isOwn()); else assert(vd2[i].isOwn());

  std::cout << "reset" << std::endl;
  vd2[3].reset(); vd2[6].reset();
  assert(vd2[3].empty());assert(vd2[6].empty());

  std::cout << "remove" << std::endl;
  auto last = std::remove_if(vd2.begin(),vd2.end(),[](PD const & p) {return p.empty();});
  for (auto i=vd2.begin(); i!=last; ++i)  std::cout << **i<< ' '; std::cout << std::endl;
  vd2.resize(last-vd2.begin());
  assert(8==vd2.size());
  for (auto i=0U; i<vd2.size(); ++i) std::cout << *vd2[i] << ' '; std::cout << std::endl;


  return 0;
}

