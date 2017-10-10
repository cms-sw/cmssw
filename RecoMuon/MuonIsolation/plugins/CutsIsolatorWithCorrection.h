#ifndef CutsIsolatorWithCorrection_H
#define CutsIsolatorWithCorrection_H

#include "RecoMuon/MuonIsolation/interface/MuIsoBaseIsolator.h"
#include "RecoMuon/MuonIsolation/interface/Cuts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class CutsIsolatorWithCorrection : public muonisolation::MuIsoBaseIsolator {
 public:
  CutsIsolatorWithCorrection(const edm::ParameterSet & par, 
			     edm::ConsumesCollector && iC);

  ResultType resultType() const override {return ISOL_BOOL_TYPE;}

  Result result(const DepositContainer& deposits, const edm::Event* = nullptr) const override {
    Result answer(ISOL_BOOL_TYPE);
    answer.valBool = false;
    // fail miserably...
    return answer;
  }

  Result result(const DepositContainer& deposits, const reco::Track& tk, const edm::Event* = nullptr) const override;
  
 private:
  double depSum(const DepositContainer& deposits, double dr, double corr) const;

  // Isolation cuts
  muonisolation::Cuts theCuts;
  muonisolation::Cuts theCutsRel;

  bool theCutAbsIso;
  bool theCutRelativeIso;
  bool theUseRhoCorrection;
  edm::EDGetTokenT<double> theRhoToken;
  double theRhoMax;
  double theRhoScaleBarrel;
  double theRhoScaleEndcap;
  double theEffAreaSFBarrel;
  double theEffAreaSFEndcap;
  bool theReturnAbsoluteSum;
  bool theReturnRelativeSum;
  bool theAndOrCuts;

};

#endif
