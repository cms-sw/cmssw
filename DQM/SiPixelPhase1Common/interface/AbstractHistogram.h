#ifndef SiPixel_AbstractHistogram_h
#define SiPixel_AbstractHistogram_h
// -*- C++ -*-
//
// Package:    SiPixelPhase1Common
// Class:      AbstractHistogram
//
// This is a spaceholder for a histogram in 0, 1, or 2 Dimensions. May or may 
// not be backed by a TH1 or similar. May not be there at all and created on 
// demand. Mainly designed as a value in std::map.
//
// Original Author:  Marcel Schneider
//

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiPixelPhase1Common/interface/GeometryInterface.h"
#include <vector>
#include <utility>
#include <cassert>

struct AbstractHistogram {

  int count = 0; // how many things where inserted already. For concat.
  MonitorElement* me = nullptr;
  TH1* th1 = nullptr;
  // This is needed for re-grouping, which happens for counters and harvesting
  // This is always an iq out of GeometryInterface::allModules
  GeometryInterface::InterestingQuantities iq_sample;

  ~AbstractHistogram() {
    // if both are set the ME should own the TH1
    if (th1 && !me) {
      //std::cout << "+++ Deleting " << th1->GetTitle() << "\n";
      delete th1;
    }
  };

};


#endif
