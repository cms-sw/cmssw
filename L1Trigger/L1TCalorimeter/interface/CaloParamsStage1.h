// CaloParamsStage1.h
// Author: R. Alex Barbieri
//
// Wrapper class for CaloParams and Et scales

#include "CondFormats/L1TObjects/interface/CaloParams.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"

#ifndef CaloParamsStage1_h
#define CaloParamsStage1_h

namespace l1t {

  class CaloParamsStage1 : public CaloParams {

  public:
    CaloParamsStage1() {}
    CaloParamsStage1(const CaloParams);
    ~CaloParamsStage1() {}

    L1CaloEtScale emScale() { return emScale_; }
    void setEmScale(L1CaloEtScale emScale) { emScale_ = emScale; }
    L1CaloEtScale jetScale() { return jetScale_; }
    void setJetScale(L1CaloEtScale jetScale) { jetScale_ = jetScale; }
    L1CaloEtScale HtMissScale() {return HtMissScale_;}
    L1CaloEtScale HfRingScale() {return HfRingScale_;}
    void setHtMissScale(L1CaloEtScale HtMissScale){HtMissScale_ = HtMissScale;}
    void setHfRingScale(L1CaloEtScale HfRingScale){HfRingScale_ = HfRingScale;}



  private:
    L1CaloEtScale emScale_;
    L1CaloEtScale jetScale_;
    L1CaloEtScale HtMissScale_;
    L1CaloEtScale HfRingScale_;

  };
}

#endif
