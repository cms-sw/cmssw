
#ifndef __LASPEAKFINDER_H
#define __LASPEAKFINDER_H


#include <utility>
#include <cmath>
#include <iostream>

#include <TH1.h>
#include <TF1.h>

#include "Alignment/LaserAlignment/src/LASModuleProfile.h"


///
/// class for fitting laser peaks
/// in a LASModuleProfile;
/// (will replace BeamProfileFitter)
///
class LASPeakFinder {

 public:
  LASPeakFinder();
  bool FindPeakIn( const LASModuleProfile&, std::pair<double,double>&, const int );

};

#endif
