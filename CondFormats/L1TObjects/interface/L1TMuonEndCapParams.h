//
// L1TMuonEndCapParams: parameters needed to calculte the EMTF algorithm
//

#ifndef l1t_L1TMuonendCapParams_h
#define l1t_L1TMuonendCapParams_h

#include <memory>
#include <iostream>
#include <vector>
#include <map>

#include "CondFormats/Serialization/interface/Serializable.h"

class L1TMuonEndCapParams {
 public:
  		
  L1TMuonEndCapParams() { PtAssignVersion_=1; firmwareVersion_=1; 
    PhiMatchWindowSt1_ = 0; PhiMatchWindowSt2_ = 0; PhiMatchWindowSt3_ = 0;  PhiMatchWindowSt4_ = 0;
  }
  ~L1TMuonEndCapParams() {}

  // FIXME MULHEARN:  this requires cleanup too, but leaving as is for now:
  unsigned PtAssignVersion_, firmwareVersion_;
  int PhiMatchWindowSt1_, PhiMatchWindowSt2_, PhiMatchWindowSt3_, PhiMatchWindowSt4_;
  
  COND_SERIALIZABLE;
};
#endif
