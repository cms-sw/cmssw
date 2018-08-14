#ifndef MuonIsolation_CaloExtractor_H
#define MuonIsolation_CaloExtractor_H

#include <string>

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

namespace muonisolation {

class CaloExtractor : public reco::isodeposit::IsoDepositExtractor {

public:

  CaloExtractor(){};
  CaloExtractor(const edm::ParameterSet& par, edm::ConsumesCollector && iC);

  ~CaloExtractor() override{}

  void fillVetos (const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackCollection & tracks) override;
  reco::IsoDeposit deposit (const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const override;

  /// Extrapolate muons to calorimeter-object positions
  static GlobalPoint MuonAtCaloPosition(const reco::Track& muon, const double bz, const GlobalPoint& endpos, bool fixVxy=false, bool fixVz=false);

private:
  // CaloTower Collection Label
  edm::EDGetTokenT<CaloTowerCollection> theCaloTowerCollectionToken;

  // Label of deposit
  std::string theDepositLabel;

  // Cone cuts and thresholds
  double theWeight_E;
  double theWeight_H;
  double theThreshold_E;
  double theThreshold_H;
  double theDR_Veto_E;
  double theDR_Veto_H;
  double theDR_Max;
  bool vertexConstraintFlag_XY;
  bool vertexConstraintFlag_Z;

  // Vector of calo Ids to veto
  std::vector<DetId> theVetoCollection;

  // Determine noise for HCAL and ECAL (take some defaults for the time being)
  double noiseEcal(const CaloTower& tower) const;
  double noiseHcal(const CaloTower& tower) const;
};

}

#endif
