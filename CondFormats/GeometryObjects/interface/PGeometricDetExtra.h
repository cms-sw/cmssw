#ifndef CondFormats_PGeometricDetExtra_h
#define CondFormats_PGeometricDetExtra_h

#include <vector>
#include <string>

class PGeometricDetExtra{

 public:
  PGeometricDetExtra() { };
  ~PGeometricDetExtra() { };

  struct Item{  
    int _geographicalID; // to be converted to DetId
    //    mutable DetId _geographicalId;
    //  std::vector< DDExpandedNode > _parents; DO NOT SAVE!
    //GeoHistory _parents;
    double _volume;
    double _density;
    double _weight;
    int    _copy;
    std::string _material;
  };

  std::vector<Item> pgdes_;

};

#endif

