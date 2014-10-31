#ifndef PTrackerParameters_h
#define PTrackerParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>

class PTrackerParameters
{
 public:
  PTrackerParameters( void ) { } 
  ~PTrackerParameters( void ) { }

  std::vector<int> pfupars;   // user parameters
  
  COND_SERIALIZABLE;
};

#endif
