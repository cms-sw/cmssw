#ifndef CondFormats_PGeometricTimingDetExtra_h
#define CondFormats_PGeometricTimingDetExtra_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <string>

class PGeometricTimingDetExtra{

 public:
  PGeometricTimingDetExtra() { };
  ~PGeometricTimingDetExtra() { };

  struct Item{  
    int geographicalId_; // to be converted to DetId
    //  std::vector< DDExpandedNode > parents_; DO NOT SAVE!
    //GeoHistory _parents;
    double volume_;
    double density_;
    double weight_;
    int    copy_;
    std::string material_;
  
  COND_SERIALIZABLE;
};

  std::vector<Item> pgdes_;


 COND_SERIALIZABLE;
};

#endif

