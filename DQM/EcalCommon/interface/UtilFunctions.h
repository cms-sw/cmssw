#ifndef DQM_ECALCOMMON_UtilFunctions_H
#define DQM_ECALCOMMON_UtilFunctions_H

/*!
  \file UtilFunctions.h
  \brief Ecal Monitor Utility functions
  \author Dongwook Jang
  \version $Revision: 1.2 $
  \date $Date: 2010/08/06 12:28:07 $
*/

#include <cmath>
#include <iostream>

#include "TH1F.h"
#include "TProfile.h"
#include "TClass.h"

class MonitorElement;

namespace ecaldqm {

  // functions implemented here are not universal in the sense that
  // the input variables are changed due to efficiency of memory usage.


  // calculated time intervals and bin locations for time varing profiles

  void calcBins(int binWidth, int divisor, long int start_time, long int last_time, long int current_time,
		long int & binDiff, long int & diff);

  // shift bins in TProfile to the right

  void shift2Right(TProfile* p, int bins);

  // shift bins in TProfile to the left

  void shift2Left(TProfile* p, int bins);

  // get mean and rms of Y values from TProfile

  void getAverageFromTProfile(TProfile* p, double& mean, double& rms);

  // get mean and rms based on two histograms' difference

  void getMeanRms(TObject* pre, TObject* cur, double& mean, double& rms); 

  TObject* cloneIt(MonitorElement* me, std::string histo); 

}

#endif
