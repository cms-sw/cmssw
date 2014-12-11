#ifndef CondFormats_EcalObjects_EcalSamplesCorrelation_HH
#define CondFormats_EcalObjects_EcalSamplesCorrelation_HH

// -*- C++ -*-
//
// Author:      Jean Fay
// Created:     Monday 24 Nov 2014
//

#include "CondFormats/Serialization/interface/Serializable.h"

#include "DataFormats/Math/interface/Matrix.h"
#include <iostream>
#include <vector>

class EcalSamplesCorrelation {
 public:
  EcalSamplesCorrelation();
  EcalSamplesCorrelation(const EcalSamplesCorrelation& aset);
  ~EcalSamplesCorrelation();

  std::vector<double> EBG12SamplesCorrelation;
  std::vector<double> EBG6SamplesCorrelation;
  std::vector<double> EBG1SamplesCorrelation;

  std::vector<double> EEG12SamplesCorrelation;
  std::vector<double> EEG6SamplesCorrelation;
  std::vector<double> EEG1SamplesCorrelation;

  void print(std::ostream& o) const;

 COND_SERIALIZABLE;
};

#endif
