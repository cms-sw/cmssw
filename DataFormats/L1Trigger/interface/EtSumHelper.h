//
// EtSumHelper:  Helper Class for Interpreting L1T EtSum output
// 

#ifndef L1T_ETSUMHELPER_H
#define L1T_ETSUMHELPER_H

#include "DataFormats/L1Trigger/interface/EtSum.h"

namespace l1t {

  class EtSumHelper{
  public:
    EtSumHelper(edm::Handle<l1t::EtSumBxCollection> & sum ):sum_(sum) {} // class assumes sum has been checked to be valid. 
    double MissingEt();
    double MissingEtPhi();
    double MissingHt();
    double MissingHtPhi();
    double TotalEt();
    double TotalHt();

  private:
    edm::Handle<l1t::EtSumBxCollection> & sum_;
  };
}

#endif

