// io ho un link a boost
// Boost1331 -> /afs/cern.ch/cms/external/lcg/external/Boost/1.33.1/slc4_ia32_gcc34/include/boost-1_33_1/
// 
// g++ -IBoost1331/ testTrackerPhiSort.cpp
//  -g -O2 -fPIC -pthread
// or
// g++4 -O3 -pthread -IBoost1331/ testTrackerPhiSort.cpp
// 
// to compile use this command:
// g++4 -O3 -pthread -I/afs/cern.ch/cms/external/lcg/external/Boost/1.33.1/slc4_ia32_gcc34/include/boost-1_33_1/ testTrackerPhiSort.cpp
//


#include "../interface/TrackerStablePhiSort.h"

#include "FakeCPP.h"

#include<cmath>
#include<ctime>


#include<vector>
#include<algorithm>
#include<iterator>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/assign/std/vector.hpp>
// for operator =+
using namespace boost::assign;

// stubs
namespace {
  
  struct XY {
    XY(double ix=0,double iy=0):x(ix),y(iy){}
    double x;
    double y;
  };
  bool operator==(XY const & a, XY const & b) {
    return a.x==b.x&&a.y==b.y;
  }

  struct GetPhi {
    typedef double result_type;
    double operator()(XY const & xy) const {
      static const double pi2 = 2.*M_PI;
      return xy.y >= 0 ?  
	std::atan2(xy.y,xy.x) :
	pi2-std::atan2(-xy.y,xy.x);
	//double phi = std::atan2(xy.y,xy.x);
	//return phi<0 ? phi+pi2 : phi;
    }
  };


  XY fromRP(double r, double phi) {
    XY xy(r*std::cos(phi), r*std::sin(phi));
    return xy;
  }
}


template<typename T>
inline void crap(T t1, T t2) {
  std::random_shuffle(t1,t2);
}

template<typename V>
void printPhi(V const & v ) {
  std::transform(v.begin(),
		 v.end(),
		 std::ostream_iterator<double>(std::cout," "), 
		 boost::bind(GetPhi(),_1)
		 );
		
  std::cout << std::endl;
}

int main() {
  cppUnit::Dump a;

  typedef std::vector<XY> Vector;

 typedef std::vector<Vector> Collection;

  Collection original, shuffled, sorted;

  // test1
  {
    // ordered....
    std::vector<double> phis; phis += 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3., 3.1, 4.2, 5.2, 6.25;
    // add an almost 0 (but not quite)
    phis += GetPhi()(XY(1.E5,-1.));
    original += Vector(phis.size());
    std::transform(phis.begin(),phis.end(),original.back().begin(),boost::bind(fromRP,2.,_1));
  }  // test1
  {
    // ordered....
    std::vector<double> phis; phis += 0.000001, 0.5, 1.0, 1.5, 2.0, 2.5, 3., 3.1, 4.2, 5.2, 6.25;
    // add an almost 0 (is zero for the algo)
    phis.insert(phis.begin(),(GetPhi()(XY(1.E10,-1.))));
    original += Vector(phis.size());
    std::transform(phis.begin(),phis.end(),original.back().begin(),boost::bind(fromRP,2.,_1));
  }		 
  // TEC tests???
  {
    // ordered....
    std::vector<double> phis; phis += 0.0, 0.5, 3., 5.2, 6.25;
    original += Vector(phis.size());
    std::transform(phis.begin(),phis.end(),original.back().begin(),boost::bind(fromRP,2.,_1));
  }		 
		 
  {
    // ordered....
    std::vector<double> phis; phis += -0.1683, -0.0561, +0.0000, +0.0561 +0.1683;
    original += Vector(phis.size());
    std::transform(phis.begin(),phis.end(),original.back().begin(),boost::bind(fromRP,2.,_1));
  }		 


  std::for_each(original.begin(),original.end(),printPhi<Vector>);

  // do the test
  shuffled.resize(original.size());
  sorted.resize(original.size());
  std::copy(original.begin(),original.end(),shuffled.begin());
  // boost::function<void(Vector::iterator,Vector::iterator)> f = &std::random_shuffle<Vector::iterator>;
  std::for_each(shuffled.begin(),shuffled.end(),
		//		boost::bind(f,
		//boost::bind(std::random_shuffle<Vector::iterator>,  no idea why does not compile....
	       		    boost::bind(crap<Vector::iterator>,
			    boost::bind<Vector::iterator>(&Vector::begin,_1),
			    boost::bind<Vector::iterator>(&Vector::end,_1)
			    )
		);
 
  
  std::copy(shuffled.begin(),shuffled.end(),sorted.begin());

  CPPUNIT_ASSERT(original!=sorted);

  double before = std::clock();
  std::for_each(sorted.begin(),sorted.end(),
		boost::bind(TrackerStablePhiSort<Vector::iterator,GetPhi>,
			    boost::bind<Vector::iterator>(&Vector::begin,_1),
			    boost::bind<Vector::iterator>(&Vector::end,_1),GetPhi()
			    )
		);

  double after = std::clock();
  std::cout << "elapsed " << (after-before)*0.01 << std::endl;

  CPPUNIT_ASSERT(original==sorted);

  std::for_each(sorted.begin(),sorted.end(),printPhi<Vector>);

  std::cout << std::endl;
  return 0;
}
