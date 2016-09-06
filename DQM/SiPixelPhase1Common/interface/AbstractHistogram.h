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

  void fill(double x, double y, double z) {
    if (me) {
      me->Fill(x, y, z);
      return;
    } else if (th1) {
      assert(!"Invalid operation on TH1");
    } else {
      assert(!"Invalid histogram. This is a problem in the HistogramManager.");
    } 
  }; 

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
  MonitorElement* me = nullptr;
  TH1* th1 = nullptr;

  // full set of metadata. _Only_ used during the booking method.
  double range_x_min = 1e12;
  double range_x_max = -1e12;
  double range_y_min = 1e12;
  double range_y_max = -1e12;
  int range_x_nbins = 0;
  int range_y_nbins = 0;
  std::string name, title, xlabel, ylabel;
  MonitorElement::Kind kind = MonitorElement::DQM_KIND_INVALID;

  ~AbstractHistogram() {
    // if both are set the ME should own the TH1
    if (th1 && !me) {
      //std::cout << "+++ Deleting " << th1->GetTitle() << "\n";
      delete th1;
    }
  };

};


#endif
