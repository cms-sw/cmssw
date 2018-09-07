#ifndef CondFormats_GeometryObjects_PMTDParameters_h
#define CondFormats_GeometryObjects_PMTDParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

class PMTDParameters
{
 public:
  PMTDParameters( void ) { } 
  ~PMTDParameters( void ) { }

  struct Item
  {
    int id_;
    std::vector<int> vpars_;
    
    COND_SERIALIZABLE;
  };

  std::vector<Item> vitems_;
  std::vector<int> vpars_;
  
  COND_SERIALIZABLE;
};

#endif
