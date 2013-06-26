#ifndef DQM_ECALCOMMON_UtilFunctions_H
#define DQM_ECALCOMMON_UtilFunctions_H

/*!
  \file UtilFunctions.h
  \brief Ecal Monitor Utility functions
  \author Dongwook Jang
  \version $Revision: 1.5 $
  \date $Date: 2011/08/12 08:25:21 $
*/

#include <cmath>
#include <iostream>

#include "TH1F.h"
#include "TProfile.h"
#include "TClass.h"

class MonitorElement;

namespace ecaldqm {

  enum Directions {
    kLeft,
    kRight
  };

  // functions implemented here are not universal in the sense that
  // the input variables are changed due to efficiency of memory usage.


  // calculated time intervals and bin locations for time varing profiles

  void calcBins(int binWidth, int divisor, long int start_time, long int last_time, long int current_time,
		long int & binDiff, long int & diff);

  // shift bins in the given direction

  void shift(TH1 *h, Directions d, int bins);

  // shift bins in histograms to the right

  void shift2Right(TH1* h, int bins);

  // shift bins in histograms to the left

  void shift2Left(TH1* h, int bins);

  // shift axis of histograms keeping the bin contents

  void shiftAxis(TH1 *h, Directions d, double shift);

  // get mean and rms of Y values from TProfile

  void getAverageFromTProfile(TProfile* p, double& mean, double& rms);

  // get mean and rms based on two histograms' difference

  void getMeanRms(TObject* pre, TObject* cur, double& mean, double& rms); 

  TObject* cloneIt(MonitorElement* me, std::string histo); 

}

#endif
