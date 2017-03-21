#ifndef Geometry_HGCalCommonData_FastTimeParameters_h
#define Geometry_HGCalCommonData_FastTimeParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <string>
#include <vector>
#include <iostream>
#include <stdint.h>

class FastTimeParameters {

public:
  
  FastTimeParameters();
  ~FastTimeParameters();

  int                      nZBarrel_;
  int                      nPhiBarrel_;
  int                      nEtaEndcap_;
  int                      nPhiEndcap_;
  std::vector<double>      geomParBarrel_;
  std::vector<double>      geomParEndcap_;

  COND_SERIALIZABLE;
};

#endif
