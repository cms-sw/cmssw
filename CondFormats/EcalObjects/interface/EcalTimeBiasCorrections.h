#ifndef CondFormats_EcalObjects_EcalTimeBiasCorrections_HH
#define CondFormats_EcalObjects_EcalTimeBiasCorrections_HH

// -*- C++ -*-
//
// Author:      Dmitrijus Bugelskis
// Created:     Thu, 14 Nov 2013 17:12:16 GMT
//

#include "CondFormats/Serialization/interface/Serializable.h"

#include "DataFormats/Math/interface/Matrix.h"
#include <iostream>
#include <vector>

class EcalTimeBiasCorrections {
public:
  EcalTimeBiasCorrections();
  EcalTimeBiasCorrections(const EcalTimeBiasCorrections& aset);
  ~EcalTimeBiasCorrections();

  // there is no need to getters/setters, just access data directly
  std::vector<float> EBTimeCorrAmplitudeBins;
  std::vector<float> EBTimeCorrShiftBins;

  std::vector<float> EETimeCorrAmplitudeBins;
  std::vector<float> EETimeCorrShiftBins;

  void print(std::ostream& o) const;

  COND_SERIALIZABLE;
};

#endif
