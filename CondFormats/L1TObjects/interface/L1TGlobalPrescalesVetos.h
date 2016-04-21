//  L1TGlobalPrescalesVetos
//
//  Table containing the entire set of prescales and masks for each L1T algorithm bit
//

#ifndef L1TGlobalPrescalesVetos_h
#define L1TGlobalPrescalesVetos_h

#include <vector>

#include "CondFormats/Serialization/interface/Serializable.h"

class L1TGlobalPrescalesVetos {
 public:
  L1TGlobalPrescalesVetos(){ version_ = 0; bxmask_default_=0; }

  unsigned int version_;
  std::vector<std::vector<int> > prescale_table_;
  int bxmask_default_;
  std::map<int, std::vector<int> > bxmask_map_;
  std::vector<int>  veto_;
  std::vector<int> exp_ints_;
  std::vector<double> exp_doubles_;

  COND_SERIALIZABLE;
};

#endif
