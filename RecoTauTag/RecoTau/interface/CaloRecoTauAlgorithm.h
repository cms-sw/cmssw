#ifndef RecoTauTag_RecoTau_CaloRecoTauAlgorithm_H
#define RecoTauTag_RecoTau_CaloRecoTauAlgorithm_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h" 
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h" 
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "RecoTauTag/TauTagTools/interface/CaloTauElementsOperators.h"
#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

#include "RecoJets/JetProducers/interface/JetMatchingTools.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class  CaloRecoTauAlgorithm  {
 public:
  CaloRecoTauAlgorithm();  
  CaloRecoTauAlgorithm(const edm::ParameterSet& iConfig);
  ~CaloRecoTauAlgorithm(){}
  void setTransientTrackBuilder(const TransientTrackBuilder*);
  void setMagneticField(const MagneticField*);
  reco::CaloTau buildCaloTau(edm::Event&,const edm::EventSetup&,const reco::CaloTauTagInfoRef&,const reco::Vertex&); 
  std::vector<DetId> mySelectedDetId_;
 private:
  std::vector<CaloTowerDetId> getCaloTowerneighbourDetIds(const CaloSubdetectorGeometry*, const CaloTowerTopology&, CaloTowerDetId);
  const TransientTrackBuilder* TransientTrackBuilder_;
  const MagneticField* MagneticField_;
  double LeadTrack_minPt_;
  double Track_minPt_;
  double IsolationTrack_minPt_;
  unsigned int IsolationTrack_minHits_;
  bool UseTrackLeadTrackDZconstraint_;
  double TrackLeadTrack_maxDZ_;
  double ECALRecHit_minEt_;
  std::string MatchingConeMetric_;
  std::string MatchingConeSizeFormula_;
  double MatchingConeSize_min_;
  double MatchingConeSize_max_;
  std::string TrackerSignalConeMetric_;
  std::string TrackerSignalConeSizeFormula_;
  double TrackerSignalConeSize_min_;
  double TrackerSignalConeSize_max_;
  std::string TrackerIsolConeMetric_;
  std::string TrackerIsolConeSizeFormula_;
  double TrackerIsolConeSize_min_;
  double TrackerIsolConeSize_max_;
  std::string ECALSignalConeMetric_;
  std::string ECALSignalConeSizeFormula_;
  double ECALSignalConeSize_min_;
  double ECALSignalConeSize_max_;
  std::string ECALIsolConeMetric_;
  std::string ECALIsolConeSizeFormula_;
  double ECALIsolConeSize_min_;
  double ECALIsolConeSize_max_;
  double AreaMetric_recoElements_maxabsEta_;
  const double chargedpi_mass_; //PDG Particle Physics Booklet, 2004

  TFormula myTrackerSignalConeSizeTFormula,myTrackerIsolConeSizeTFormula, myECALSignalConeSizeTFormula, myECALIsolConeSizeTFormula,myMatchingConeSizeTFormula; 
  
edm::InputTag EBRecHitsLabel_,EERecHitsLabel_,ESRecHitsLabel_; 

};
#endif 

