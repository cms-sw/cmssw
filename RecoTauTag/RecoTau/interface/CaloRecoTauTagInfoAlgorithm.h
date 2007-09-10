#ifndef RecoTauTag_RecoTau_CaloRecoTauTagInfoAlgorithm_H
#define RecoTauTag_RecoTau_CaloRecoTauTagInfoAlgorithm_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
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
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

using namespace std;
using namespace reco;
using namespace edm;

class  CaloRecoTauTagInfoAlgorithm  {
 public:
  CaloRecoTauTagInfoAlgorithm() : chargedpi_mass_(0.13957018){}  
  CaloRecoTauTagInfoAlgorithm(const ParameterSet& parameters);
  ~CaloRecoTauTagInfoAlgorithm(){}
  CaloTauTagInfo buildCaloTauTagInfo(Event&,const EventSetup&,const CaloJetRef&,const TrackRefVector&,const Vertex&); 
 private:  
  vector<pair<math::XYZPoint,float> > getPositionAndEnergyEcalRecHits(Event&,const EventSetup&,const CaloJetRef&);
  BasicClusterRefVector getNeutralEcalBasicClusters(Event&,const EventSetup& theEventSetup,const CaloJetRef&,const TrackRefVector&,float theECALBasicClustersAroundCaloJet_DRConeSize,float theECALBasicClusterminE,float theECALBasicClusterpropagTrack_matchingDRConeSize);
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
  double ECALBasicClustersAroundCaloJet_DRConeSize_;
  double ECALBasicClusterminE_;
  double ECALBasicClusterpropagTrack_matchingDRConeSize_;
  //
  const double chargedpi_mass_;
};
#endif 

