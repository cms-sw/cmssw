#ifndef MuonIsolation_CaloExtractor_H
#define MuonIsolation_CaloExtractor_H

#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "Geometry/Vector/interface/GlobalPoint.h"

namespace muonisolation {

class CaloExtractor : public MuIsoExtractor {

public:

  CaloExtractor() { }
  CaloExtractor(const edm::ParameterSet& par);

  virtual ~CaloExtractor(){}

  virtual std::vector<reco::MuIsoDeposit> deposits( const edm::Event & ev, 
      const edm::EventSetup & evSetup, const reco::Track & track, 
      const std::vector<muonisolation::Direction> & vetoDirs, double coneSize) const; 

  /// Extrapolate muons to calorimeter-object positions
  GlobalPoint MuonAtCaloPosition(const reco::Track& muon, const GlobalPoint& endpos, bool fixVxy=false, bool fixVz=false) const;
  
 private:
private:
  
  void fillDeposits( reco::MuIsoDeposit & ecaldep, reco::MuIsoDeposit & hcaldep, 
      const reco::Track& mu, const edm::Event& event, const edm::EventSetup& eventSetup) const;

private:
  double theDiff_r, theDiff_z, theDR_Match, theDR_Veto;
  std::string theCaloTowerCollectionLabel; // CaloTower Collection Label

  // Cone cuts and thresholds
  double theThreshold_E;
  double theThreshold_H;
  double theDR_Veto_E;
  double theDR_Veto_H;
  double theDR_Max;
  bool vertexConstraintFlag_XY;
  bool vertexConstraintFlag_Z;

  // Determine noise for HCAL and ECAL (take some defaults for the time being)
  double noiseEcal(const CaloTower& tower) const;
  double noiseHcal(const CaloTower& tower) const;

  // Function to ensure that phi and theta are in range
  double PhiInRange(const double& phi) const;

  // DeltaR function
  template <class T, class U> double deltaR(const T& t, const U& u) const;
};

}

#endif
