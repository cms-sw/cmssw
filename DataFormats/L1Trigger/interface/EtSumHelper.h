//
// EtSumHelper:  Helper Class for Interpreting L1T EtSum output
// 

#ifndef DataFormats_L1Trigger_EtSumHelper_h
#define DataFormats_L1Trigger_EtSumHelper_h

#include "DataFormats/L1Trigger/interface/EtSum.h"

namespace l1t {

  class EtSumHelper{
  public:
    EtSumHelper(const edm::Handle<l1t::EtSumBxCollection> & sum ):sum_(sum) {} // class assumes sum has been checked to be valid. 
    double MissingEt() const;
    double MissingEtPhi() const;
    double MissingHt() const;
    double MissingHtPhi() const;
    double TotalEt() const;
    double TotalHt() const;

  private:
    const edm::Handle<l1t::EtSumBxCollection> & sum_;
  };
}

#endif

