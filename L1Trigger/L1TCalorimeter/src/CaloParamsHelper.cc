// CaloParamsHelper.cc
// Author: R. Alex Barbieri
//
// Wrapper class for CaloParams and Et scales

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {
  CaloParamsHelper::CaloParamsHelper(const CaloParams p) : CaloParams(p) {};

  int CaloParamsHelper::etSumEtaMin(unsigned isum) const {
    if (etSumEtaMin_.size()>isum) return etSumEtaMin_.at(isum);
    else return 0;
  }

  int CaloParamsHelper::etSumEtaMax(unsigned isum) const {
    if (etSumEtaMax_.size()>isum) return etSumEtaMax_.at(isum);
    else return 0;
  }

  double CaloParamsHelper::etSumEtThreshold(unsigned isum) const {
    if (etSumEtThreshold_.size()>isum) return etSumEtThreshold_.at(isum);
    else return 0.;
  }

  void CaloParamsHelper::setEtSumEtaMin(unsigned isum, int eta) {
    if (etSumEtaMin_.size()<=isum) etSumEtaMin_.resize(isum+1);
    etSumEtaMin_.at(isum) = eta;
  }

  void CaloParamsHelper::setEtSumEtaMax(unsigned isum, int eta) {
    if (etSumEtaMax_.size()<=isum) etSumEtaMax_.resize(isum+1);
    etSumEtaMax_.at(isum) = eta;
  }

  void CaloParamsHelper::setEtSumEtThreshold(unsigned isum, double thresh) {
    if (etSumEtThreshold_.size()<=isum) etSumEtThreshold_.resize(isum+1);
    etSumEtThreshold_.at(isum) = thresh;
  }
}
