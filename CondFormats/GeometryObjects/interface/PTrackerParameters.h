#ifndef CondFormats_GeometryObjects_PTrackerParameters_h
#define CondFormats_GeometryObjects_PTrackerParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

class PTrackerParameters
{
 public:
  PTrackerParameters( void ) { } 
  ~PTrackerParameters( void ) { }

  struct Item
  {
    int id;
    std::vector<int> vpars;
    
    COND_SERIALIZABLE;
  };

  std::vector<Item> vitems;
  std::vector<int> vpars;
  
  COND_SERIALIZABLE;
};

#endif
