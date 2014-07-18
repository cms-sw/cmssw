
#include "GsfElectronFull5x5Filler.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"
#include "RecoParticleFlow/PFProducer/interface/GsfElectronEqual.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <iostream>
#include <string>

using namespace reco;

GsfElectronFull5x5Filler::GsfElectronFull5x5Filler( const edm::ParameterSet & cfg )
 {
   _source = consumes<reco::GsfElectronCollection>(cfg.getParameter<edm::InputTag>("source"));
   
   ElectronHcalHelper::Configuration hcalCfg, hcalCfgPflow;
   
   hcalCfg.hOverEConeSize = cfg.getParameter<double>("hOverEConeSize") ;
   if (hcalCfg.hOverEConeSize>0)
     {
       hcalCfg.useTowers = true ;
       hcalCfg.hcalTowers = 
	 consumes<CaloTowerCollection>(cfg.getParameter<edm::InputTag>("hcalTowers")) ;
       hcalCfg.hOverEPtMin = cfg.getParameter<double>("hOverEPtMin") ;
     }
   hcalCfgPflow.hOverEConeSize = cfg.getParameter<double>("hOverEConeSizePflow") ;
   if (hcalCfgPflow.hOverEConeSize>0)
     {
       hcalCfgPflow.useTowers = true ;
       hcalCfgPflow.hcalTowers = 
	 consumes<CaloTowerCollection>(cfg.getParameter<edm::InputTag>("hcalTowers")) ;
       hcalCfgPflow.hOverEPtMin = cfg.getParameter<double>("hOverEPtMinPflow") ;
     }
   _hcalHelper.reset(new ElectronHcalHelper(hcalCfg));
   _hcalHelperPflow.reset(new ElectronHcalHelper(hcalCfg));
   
   _ebRecHitsToken = consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("barrelRecHitCollectionTag"));
   _eeRecHitsToken = consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("endcapRecHitCollectionTag"));
   
   produces<reco::GsfElectronCollection >();
 }

GsfElectronFull5x5Filler::~GsfElectronFull5x5Filler()
 {}

// ------------ method called to produce the data  ------------
void GsfElectronFull5x5Filler::produce( edm::Event & event, const edm::EventSetup & setup )
 {
   std::auto_ptr<reco::GsfElectronCollection> out(new reco::GsfElectronCollection);
   
   edm::Handle<reco::GsfElectronCollection> eles;
   event.getByToken(_source,eles);
   
   event.getByToken(_ebRecHitsToken,_ebRecHits);
   event.getByToken(_eeRecHitsToken,_eeRecHits);

   for( const auto& ele : *eles ) {
     reco::GsfElectron temp(ele);
     reco::GsfElectron::ShowerShape full5x5_ss;
     calculateShowerShape_full5x5(temp.superCluster(),true,full5x5_ss);
     temp.full5x5_setShowerShape(full5x5_ss);
     out->push_back(temp);
   }
   
   event.put(out);
 }

void GsfElectronFull5x5Filler::beginLuminosityBlock(edm::LuminosityBlock const& lb, 
						 edm::EventSetup const& es) {
  edm::ESHandle<CaloGeometry> caloGeom ;
  edm::ESHandle<CaloTopology> caloTopo ;
  es.get<CaloGeometryRecord>().get(caloGeom);
  es.get<CaloTopologyRecord>().get(caloTopo);
  _geometry = caloGeom.product();
  _topology = caloTopo.product();
}

void GsfElectronFull5x5Filler::calculateShowerShape_full5x5( const reco::SuperClusterRef & theClus, bool pflow, reco::GsfElectron::ShowerShape & showerShape )
 {
  const reco::CaloCluster & seedCluster = *(theClus->seed()) ;
  // temporary, till CaloCluster->seed() is made available
  DetId seedXtalId = seedCluster.hitsAndFractions()[0].first ;
  int detector = seedXtalId.subdetId() ;
  
  const EcalRecHitCollection * recHits = 0 ;
  if (detector==EcalBarrel)
   {
    recHits = _ebRecHits.product() ;
   }
  else
   {
    recHits = _eeRecHits.product() ;
   }

  std::vector<float> covariances = noZS::EcalClusterTools::covariances(seedCluster,recHits,_topology,_geometry) ;
  std::vector<float> localCovariances = noZS::EcalClusterTools::localCovariances(seedCluster,recHits,_topology) ;
  showerShape.sigmaEtaEta = sqrt(covariances[0]) ;
  showerShape.sigmaIetaIeta = sqrt(localCovariances[0]) ;
  if (!edm::isNotFinite(localCovariances[2])) showerShape.sigmaIphiIphi = sqrt(localCovariances[2]) ;
  showerShape.e1x5 = noZS::EcalClusterTools::e1x5(seedCluster,recHits,_topology)  ;
  showerShape.e2x5Max = noZS::EcalClusterTools::e2x5Max(seedCluster,recHits,_topology)  ;
  showerShape.e5x5 = noZS::EcalClusterTools::e5x5(seedCluster,recHits,_topology) ;
  showerShape.r9 = noZS::EcalClusterTools::e3x3(seedCluster,recHits,_topology)/theClus->rawEnergy() ;

  if (pflow)
   {
    showerShape.hcalDepth1OverEcal = _hcalHelperPflow->hcalESumDepth1(*theClus)/theClus->energy() ;
    showerShape.hcalDepth2OverEcal = _hcalHelperPflow->hcalESumDepth2(*theClus)/theClus->energy() ;
    showerShape.hcalTowersBehindClusters = _hcalHelperPflow->hcalTowersBehindClusters(*theClus) ;
    showerShape.hcalDepth1OverEcalBc = _hcalHelperPflow->hcalESumDepth1BehindClusters(showerShape.hcalTowersBehindClusters)/showerShape.e5x5 ;
    showerShape.hcalDepth2OverEcalBc = _hcalHelperPflow->hcalESumDepth2BehindClusters(showerShape.hcalTowersBehindClusters)/showerShape.e5x5 ;
   }
  else
   {
    showerShape.hcalDepth1OverEcal = _hcalHelper->hcalESumDepth1(*theClus)/theClus->energy() ;
    showerShape.hcalDepth2OverEcal = _hcalHelper->hcalESumDepth2(*theClus)/theClus->energy() ;
    showerShape.hcalTowersBehindClusters = _hcalHelper->hcalTowersBehindClusters(*theClus) ;
    showerShape.hcalDepth1OverEcalBc = _hcalHelper->hcalESumDepth1BehindClusters(showerShape.hcalTowersBehindClusters)/showerShape.e5x5 ;
    showerShape.hcalDepth2OverEcalBc = _hcalHelper->hcalESumDepth2BehindClusters(showerShape.hcalTowersBehindClusters)/showerShape.e5x5 ;
   }
 }
