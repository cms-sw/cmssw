#include "Geometry/TrackerNumberingBuilder/interface/trackerStablePhiSort.h"
#include "FakeCPP.h"

#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

namespace {

  struct XY {
    XY(double ix=0.,double iy=0.):x(ix),y(iy){}
    double x;
    double y;
  };
  bool operator==(XY const & a, XY const & b) {
    return a.x==b.x&&a.y==b.y;
  }

  double getPhi(XY const & xy) {
    static const double pi2 = 2.*M_PI;
    return xy.y >= 0 ? std::atan2(xy.y,xy.x) : pi2-std::atan2(-xy.y,xy.x);
  }

  template<typename V>
  void printPhi(V const & v ) {
    for(auto const& x : v) std::cout << getPhi(x) << " ";
    std::cout << std::endl;
  }

}

int main()
{
  cppUnit::Dump a;

  using Collection =  std::vector<std::vector<XY>>;

  Collection original;
  Collection shuffled;
  Collection sorted;

  auto fromRP = [](double phi){ return XY(2.*std::cos(phi), 2.*std::sin(phi)); };

  // test1
  {
    // ordered.... last element is almost 0 (but not quite)
    std::vector<double> phis {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3., 3.1, 4.2, 5.2, 6.25, getPhi(XY(1.E5,-1.))};
    original.emplace_back(phis.size());
    std::transform(phis.begin(),phis.end(),original.back().begin(),fromRP);
  }  // test1
  {
    // ordered.... first element is almost 0 (is zero for the algo)
    std::vector<double> phis {getPhi(XY(1.E10,-1.)), 0.000001, 0.5, 1.0, 1.5, 2.0, 2.5, 3., 3.1, 4.2, 5.2, 6.25, };
    original.emplace_back(phis.size());
    std::transform(phis.begin(),phis.end(),original.back().begin(),fromRP);
  }
  // TEC tests???
  {
    // ordered....
    std::vector<double> phis {0.0, 0.5, 3., 5.2, 6.25};
    original.emplace_back(phis.size());
    std::transform(phis.begin(),phis.end(),original.back().begin(),fromRP);
  }
  {
    // ordered....
    std::vector<double> phis {-0.1683, -0.0561, +0.0000, +0.0561, +0.1683};
    original.emplace_back(phis.size());
    std::transform(phis.begin(),phis.end(),original.back().begin(),fromRP);
  }


  for(auto const& v : original) printPhi(v);

  // do the test
  shuffled.resize(original.size());
  sorted.resize(original.size());
  std::copy(original.begin(),original.end(),shuffled.begin());
  auto rng = std::default_random_engine {};
  for(auto& v : shuffled) std::shuffle(v.begin(), v.end(), rng);

  std::copy(shuffled.begin(),shuffled.end(),sorted.begin());

  CPPUNIT_ASSERT(original!=sorted);

  double before = std::clock();
  for(auto& v : sorted) trackerStablePhiSort(v.begin(), v.end(), getPhi);

  double after = std::clock();
  std::cout << "elapsed " << (after-before)*0.01 << std::endl;

  CPPUNIT_ASSERT(original==sorted);

  for(auto const& v : sorted) printPhi(v);
  std::cout << std::endl;

  return 0;
}
