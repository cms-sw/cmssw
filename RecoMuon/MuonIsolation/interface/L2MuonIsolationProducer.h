#ifndef RecoMuon_L2MuonIsolationProducer_H
#define RecoMuon_L2MuonIsolationProducer_H

/**  \class L2MuonIsolationProducer
 * 
 *   L2 HLT muon producer:
 *
 *   \author  J.Alcaraz
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

#include "Geometry/Vector/interface/GlobalPoint.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class L2MuonIsolationProducer : public edm::EDProducer {

 public:

  /// constructor with config
  L2MuonIsolationProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L2MuonIsolationProducer(); 
  
  /// Produce isolation maps
  virtual void produce(edm::Event&, const edm::EventSetup&);
  // ex virtual void reconstruct();

  /// Find calorimeter deposits around each muon
  bool fillDeposits(reco::MuIsoDeposit& depE, reco::MuIsoDeposit& depH, const reco::Track& muon, const edm::Event&, const edm::EventSetup&);

  /// Extrapolate muons to calorimeter-object positions
  GlobalPoint MuonAtCaloPosition(const reco::Track& muon, const GlobalPoint& endpos, bool fixVxy=false, bool fixVz=false) const;
  
 private:
  
  // Muon track Collection Label
  std::string theSACollectionLabel;

  // CaloTowers Collection Label
  std::string theCaloTowerCollectionLabel;

  // Cone cuts and thresholds
  double theThreshold_E;
  double theThreshold_H;
  double theDR_Veto_E;
  double theDR_Veto_H;
  double theDR_Max;
  bool vertexConstraintFlag_XY;
  bool vertexConstraintFlag_Z;

  std::vector<double> coneCuts_;
  std::vector<double> edepCuts_;
  std::vector<double> etaBounds_;
  double ecalWeight_;

  // Determine noise for HCAL and ECAL (take some defaults for the time being)
  double noiseEcal(const CaloTower& tower) const;
  double noiseHcal(const CaloTower& tower) const;

  // Calorimeter geometry
  edm::ESHandle<CaloGeometry> caloGeom;

  // Function to ensure that phi and theta are in range
  double PhiInRange(const double& phi) const;

  // DeltaR function
  template <class T, class U> double deltaR(const T& t, const U& u) const;
};

#endif
