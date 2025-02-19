#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalShapeIntegrator.h"
#include <algorithm> // for "max","min"
#include <math.h>
#include <iostream>
#include <boost/scoped_ptr.hpp>

// Function generates a lookup map for a passed-in function (via templated object algoObject,
// which must contain method "calcpair" that spits out (x,y) pair from a type float seed.
// Each map y-value is separated from the previous value by a programmable fractional error
// relative to the previous value.
//
// Currently this function coded for only monotonically increasing or
// decreasing functions...

template <class T_algo>
void genlkupmap(double smin,
                double smax,
                double max_fracerror,
                double min_sstep,
                T_algo& algoObject,
                std::map<double,double>& m_ylookup)
{
  std::pair<double,double> thisxy, lastxy, laststoredxy;
  std::pair<double,double> minxy = algoObject.calcpair(smin);
  std::pair<double,double> maxxy = algoObject.calcpair(smax);
  
  double slope = maxxy.second - minxy.second;
  slope = (slope >= 0.0) ? 1.0 : -1.0;

  double sstep = min_sstep;

  for (double s=smin; lastxy.first<smax; s += sstep) {

    thisxy = algoObject.calcpair(s);

    double fracerror  = slope*(thisxy.second - laststoredxy.second)/ thisxy.second;
    double fracchange = slope*(thisxy.second - lastxy.second)/ thisxy.second;

    bool store_cur_pair  = false;
    bool store_prev_pair = false;

#if 0
    char str[80];
    sprintf(str, "%7.1f %7.1f (%8.3f %8.4f) %8.5f %8.5f",
	    s, sstep, thisxy.first, thisxy.second, fracerror, fracchange);
    cout << str;
#endif

    if (s == smin) {
      store_cur_pair = true;
    }
    else if ((fracerror > 2*max_fracerror) ||
             (thisxy.first > smax) ) {
      if (sstep > min_sstep) {
        // possibly overstepped the next entry, back up and reduce the step size
        s -= sstep;
        sstep = std::max(0.5*sstep, min_sstep);
#if 0
	cout << endl;
#endif
        continue;
      }
      else if (lastxy.second == laststoredxy.second) {
        store_cur_pair = true;

        // current step size is too big for the max allowed fractional error,
        // store current value and issue a warning.
        //
	//	edm::LogWarning("HcalPulseContainmentCorrection::genlkupmap") << " fractional error max exceeded";
      }
      else {
        store_prev_pair = true;

        // re-evaluate current yval with prev yval.
        fracerror = slope*(thisxy.second - lastxy.second)/ thisxy.second;

        if (fracerror > 2*max_fracerror) {
          store_cur_pair = true;

          // current step size is too big for the max allowed fractional error,
          // store current value and issue a warning.
          //
	  //	  edm::LogWarning("HcalPulseContainmentCorrection::genlkupmap") << " fractional error max exceeded";
        }
      }
    }
    else if ((fracerror  < 1.9*max_fracerror) &&
             (fracchange < 0.1*max_fracerror) &&
             (thisxy.first < 0.99 * smax) ) {
      // adapt step size to reduce iterations
      sstep *= 2.0;
    }

    if (thisxy.first > smax)
      store_cur_pair = true;
      
    if (store_prev_pair) {
      m_ylookup[lastxy.first] = lastxy.second;
      laststoredxy            = lastxy;
    }
    if (store_cur_pair) {
      m_ylookup[thisxy.first] = thisxy.second;
      laststoredxy            = thisxy;
    }

    lastxy = thisxy;
    
#if 0
    sprintf(str, " %c %c",
	    store_cur_pair ? 'C' : ' ',
	    store_prev_pair ? 'P' : ' ');
    cout << str << endl;
#endif

  }

}

//======================================================================
// Fixed Phase mode amplitude correction factor generation routines:

template <class S> 
class RecoFCcorFactorAlgo {
public:
  RecoFCcorFactorAlgo(int    num_samples,
                      double fixedphase_ns);
  std::pair<double,double> calcpair(double);
private:
  double fixedphasens_;
  double integrationwindowns_;
  double time0shiftns_;
  S shape_;
  const boost::scoped_ptr<HcalShapeIntegrator> integrator_;
};

