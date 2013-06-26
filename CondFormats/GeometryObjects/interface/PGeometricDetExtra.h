#ifndef CondFormats_PGeometricDetExtra_h
#define CondFormats_PGeometricDetExtra_h

#include <vector>
#include <string>

class PGeometricDetExtra{

 public:
  PGeometricDetExtra() { };
  ~PGeometricDetExtra() { };

  struct Item{  
    int _geographicalId; // to be converted to DetId
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

