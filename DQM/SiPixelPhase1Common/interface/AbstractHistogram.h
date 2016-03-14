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
// Original Author:  Marcel Schenider
//

#include "DQMServices/Core/interface/MonitorElement.h"
#include <vector>
#include <utility>
#include <cassert>

struct AbstractHistogram {

  void fill(double x, double y) {
    assert(!vec1d);
    count++;
    if (me) {
      me->Fill(x, y);
      return;
    } 
    if (!vec2d) {
      vec2d = new std::vector<std::pair<double,double>>();
    }
    vec2d->push_back(std::make_pair(x, y));
  }; 
  
  void fill(double x) {
    assert(!vec2d);
    count++;
    if (me) {
      me->Fill(x);
      return;
    } 
    if (!vec1d) {
      vec1d = new std::vector<double>();
    }
    vec1d->push_back(x);
  };
  
  void fill() {
    assert(!me  && !vec1d && !vec2d);
    count++;
  };
  
  int count = 0;
  MonitorElement* me = nullptr;
  std::vector<double> *vec1d = nullptr;
  std::vector<std::pair<double, double>> *vec2d = nullptr;

  ~AbstractHistogram() {
    if (vec1d) delete vec1d;
    if (vec2d) delete vec2d;
  };

};


#endif
