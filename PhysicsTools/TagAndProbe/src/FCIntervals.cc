#include "PhysicsTools/TagAndProbe/interface/FCIntervals.hh"
#include <boost/math/special_functions/beta.hpp> 
#include <boost/math/tools/minima.hpp> // brent
#include <boost/bind.hpp>

#define CONF_LEVEL 0.683

namespace FCIntervals {
  
  double Beta_ab(const double a, const double b, const double k, const double N) {
    // Calculates the fraction of the area under the
    // curve x^k*(1-x)^(N-k) between x=a and x=b
    
    if (a == b) return 0;    // don't bother integrating over zero range
    double c1 = k+1;
    double c2 = N-k+1;
    return boost::math::ibeta(c1, c2, b) - boost::math::ibeta(c1, c2, a);
  }
  
  double SearchUpper (const double low, const double pass, const double total) {
    // Integrates the binomial distribution with
    // parameters k,N, and determines what is the upper edge of the
    // integration region which starts at low which contains probability
    // content c. If an upper limit is found, the value is returned. If no
    // solution is found, -1 is returned.
    // check to see if there is any solution by verifying that the integral up
    // to the maximum upper limit (1) is greater than c
    
    double integral = Beta_ab(low, 1.0, pass, total);
    if (integral == CONF_LEVEL) return 1.0;    // lucky -- this is the solution
    if (integral < CONF_LEVEL) return -1.0;    // no solution exists
    double too_high = 1.0;            // upper edge estimate
    double too_low = low;
    double test;
    
    // use a bracket-and-bisect search
    // LM: looping 20 times might be not enough to get an accurate precision.
    // see for example bug https://savannah.cern.ch/bugs/?30246
    // now break loop when difference is less than 1E-15
    // t.b.d: use directly the beta distribution quantile
    
    for (int loop=0; loop<50; loop++) {
      test = 0.5*(too_low + too_high);
      integral = Beta_ab(low, test, pass, total);
      if (integral > CONF_LEVEL)  too_high = test;
      else too_low = test;
      if ( fabs(integral - CONF_LEVEL) <= 1.E-15) break;
    }
    return test;
  }
  
  double SearchLower(const  double high, const double pass, const double total ) {
    // Integrates the binomial distribution with
    // parameters k,N, and determines what is the lower edge of the
    // integration region which ends at high, and which contains
    // probability content c. If a lower limit is found, the value is
    // returned. If no solution is found, the -1 is returned.
    // check to see if there is any solution by verifying that the integral down
    // to the minimum lower limit (0) is greater than c
    
    double integral = Beta_ab(0.0, high, pass, total);
    if (integral == CONF_LEVEL) return 0.0;      // lucky -- this is the solution
    if (integral < CONF_LEVEL) return -1.0;      // no solution exists
    double too_low = 0.0;               // lower edge estimate
    double too_high = high;
    double test;
    
    // use a bracket-and-bisect search
    // LM: looping 20 times might be not enough to get an accurate precision.
    // see for example bug https://savannah.cern.ch/bugs/?30246
    // now break loop when difference is less than 1E-15
    // t.b.d: use directly the beta distribution quantile
    
    for (int loop=0; loop<50; loop++) {
      test = 0.5*(too_high + too_low);
      integral = Beta_ab(test, high, pass, total);
      if (integral > CONF_LEVEL)  too_low = test;
      else too_high = test;
      if ( fabs(integral - CONF_LEVEL) <= 1.E-15) break;
    }
    return test;
  }
  
  double Interval (const double low, const double pass, const double total){
    // Return the length of the interval starting at low
    // that contains CONFLEVEL of the x^k*(1-x)^(n-k)
    // distribution.
    // If there is no sufficient interval starting at low, we return 2.0
    
    double high = SearchUpper (low, pass, total);
    if (high == -1.0) return 2.0; //  so that this won't be the shortest interval
    return (high - low);
  }
  
  
  void Efficiency(const double pass, const double total,
			       double &mode, double &lowErr, double &highErr) {
    // Adapted from TGraphAsymmErrors::Efficiency to deal with weighted trees.
    //If there are no entries, then we know nothing, thus return the prior...
    
    // Error handling.
    if (0==total) {
      mode = .5; lowErr = 0.0; highErr = 1.0;
      return;
    }
    
    mode = pass/total;
    
    double low_edge;
    double high_edge; 
    std::pair<double, double> result; 
    
    // Check extremes of interval; pass == 0, pass == total.
    if (pass == 0) {
      low_edge = 0.0;
      high_edge = SearchUpper(low_edge, pass, total);
    } else if (pass == total) {
      high_edge = 1.0;
      low_edge = SearchLower(high_edge, pass, total);
    } else { // Search for minima.
      result = 
	boost::math::tools::brent_find_minima (boost::bind(Interval, _1, pass, total), 
					       0.0, 1.0, 6);
      high_edge = result.first + result.second;
    }
    
    // return output
    lowErr = mode - result.second;
    highErr = high_edge - mode;
  }
}
