#ifndef EgammaIsolationProducers_EgammaRecHitIsolation_h
#define EgammaIsolationProducers_EgammaRecHitIsolation_h
//*****************************************************************************
// File:      EgammaRecHitIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, adapted from EgammaHcalIsolation by S. Harper
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************

//C++ includes
#include <vector>
#include <functional>

//CMSSW includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class EgammaRecHitIsolation {
public:
  //constructors
  EgammaRecHitIsolation(double extRadius,
                        double intRadius,
                        double etaSlice,
                        double etLow,
                        double eLow,
                        edm::ESHandle<CaloGeometry>,
                        const EcalRecHitCollection&,
                        const EcalSeverityLevelAlgo*,
                        DetId::Detector detector);

  double getEtSum(const reco::Candidate* emObject) const { return getSum_(emObject, true); }
  double getEnergySum(const reco::Candidate* emObject) const { return getSum_(emObject, false); }

  double getEtSum(const reco::SuperCluster* emObject) const { return getSum_(emObject, true); }
  double getEnergySum(const reco::SuperCluster* emObject) const { return getSum_(emObject, false); }

  void setUseNumCrystals(bool b = true) { useNumCrystals_ = b; }
  void setVetoClustered(bool b = true) { vetoClustered_ = b; }
  void doSeverityChecks(const EcalRecHitCollection* const recHits, const std::vector<int>& v) {
    ecalBarHits_ = recHits;
    severitiesexcl_.clear();
    severitiesexcl_.insert(severitiesexcl_.begin(), v.begin(), v.end());
    std::sort(severitiesexcl_.begin(), severitiesexcl_.end());
  }

  void doFlagChecks(const std::vector<int>& v) {
    flags_.clear();
    flags_.insert(flags_.begin(), v.begin(), v.end());
    std::sort(flags_.begin(), flags_.end());
  }

  //destructor
  ~EgammaRecHitIsolation();

private:
  double getSum_(const reco::Candidate*, bool returnEt) const;
  double getSum_(const reco::SuperCluster*, bool returnEt) const;

  double extRadius_;
  double intRadius_;
  double etaSlice_;
  double etLow_;
  double eLow_;

  edm::ESHandle<CaloGeometry> theCaloGeom_;
  const EcalRecHitCollection& caloHits_;
  const EcalSeverityLevelAlgo* sevLevel_;

  bool useNumCrystals_;
  bool vetoClustered_;
  const EcalRecHitCollection* ecalBarHits_;
  std::vector<int> severitiesexcl_;
  std::vector<int> flags_;

  const CaloSubdetectorGeometry* subdet_[2];  // barrel+endcap
};

#endif
