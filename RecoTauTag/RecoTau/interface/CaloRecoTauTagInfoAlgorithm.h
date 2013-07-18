#ifndef RecoTauTag_RecoTau_CaloRecoTauTagInfoAlgorithm_H
#define RecoTauTag_RecoTau_CaloRecoTauTagInfoAlgorithm_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h" 
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h" 
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

#include "DataFormats/TrackReco/interface/TrackBase.h"

#include "DataFormats/JetReco/interface/JetCollection.h"

class  CaloRecoTauTagInfoAlgorithm  {
 public:
  CaloRecoTauTagInfoAlgorithm(){}  
  CaloRecoTauTagInfoAlgorithm(const edm::ParameterSet& parameters);
  ~CaloRecoTauTagInfoAlgorithm(){}
  reco::CaloTauTagInfo buildCaloTauTagInfo(edm::Event&,const edm::EventSetup&,const reco::CaloJetRef&,const reco::TrackRefVector&,const reco::Vertex&); 
  reco::CaloTauTagInfo buildCaloTauTagInfo(edm::Event&,const edm::EventSetup&,const reco::JetBaseRef&,const reco::TrackRefVector&,const reco::Vertex&);
  std::vector<DetId> getVectorDetId(const reco::CaloJetRef&);

 private:  
  //  std::vector<std::pair<math::XYZPoint,float> > getPositionAndEnergyEcalRecHits(edm::Event&,const edm::EventSetup&,const reco::CaloJetRef&);

  std::vector<reco::BasicClusterRef> getNeutralEcalBasicClusters(edm::Event&,const edm::EventSetup& theEventSetup,const reco::CaloJetRef&,const reco::TrackRefVector&,float theECALBasicClustersAroundCaloJet_DRConeSize,float theECALBasicClusterminE,float theECALBasicClusterpropagTrack_matchingDRConeSize);
  reco::TrackRefVector filterTracksByQualityBit(const reco::TrackRefVector& tracks, reco::TrackBase::TrackQuality quality) const;
  //
  double tkminPt_;
  int tkminPixelHitsn_;
  int tkminTrackerHitsn_;
  double tkmaxipt_;
  double tkmaxChi2_;
  // 
  bool UsePVconstraint_;
  double tkPVmaxDZ_;
  //
  bool UseTrackQuality_;
  reco::TrackBase::TrackQuality tkQuality_;
  //
  double ECALBasicClustersAroundCaloJet_DRConeSize_;
  double ECALBasicClusterminE_;
  double ECALBasicClusterpropagTrack_matchingDRConeSize_;

  // 
  edm::InputTag EBRecHitsLabel_,EERecHitsLabel_,ESRecHitsLabel_; 
  edm::InputTag BarrelBasicClusters_,EndcapBasicClusters_;
};
#endif 

