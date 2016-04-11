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
#include <vector>
#include <utility>
#include <cassert>

struct AbstractHistogram {

  void fill(double x, double y) {
    if (me) {
      me->Fill(x, y);
      return;
    } else if (th1) {
      th1->Fill(x, y);
    } else {
      assert(!"Invalid histogram. This is a problem in the HistogramManager.");
    } 
  }; 
  
  void fill(double x) {
    if (me) {
      me->Fill(x);
      return;
    } else if (th1) {
      th1->Fill(x);
    } else {
      assert(!"Invalid histogram. This is a problem in the HistogramManager.");
    }
  };
  
  int count = 0; // how many things where inserted already. For concat.
  bool is0d = false; // for a 0D histogram we still have a TH1D, with 1 bin and metadata on the y-Axis.
  MonitorElement* me = nullptr;
  TH1* th1 = nullptr;

  ~AbstractHistogram() {
    // if both are set the ME should own the TH1
    if (th1 && !me) delete th1;
  };

};


#endif
