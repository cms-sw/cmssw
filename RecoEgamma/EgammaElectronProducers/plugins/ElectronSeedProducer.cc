// -*- C++ -*-
//
// Package:    ElectronProducers
// Class:      ElectronSeedProducer
//
/**\class ElectronSeedProducer RecoEgamma/ElectronProducers/src/ElectronSeedProducer.cc

 Description: EDProducer of ElectronSeed objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//

#include <vector>

#include "ElectronSeedProducer.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronSeedGenerator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/SeedFilter.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include <string>

using namespace reco ;

ElectronSeedProducer::ElectronSeedProducer( const edm::ParameterSet& iConfig )
 : //conf_(iConfig),
   applyHOverECut_(true), hcalHelper_(nullptr),
   caloGeom_(nullptr), caloGeomCacheId_(0), caloTopo_(nullptr), caloTopoCacheId_(0)
 {
  conf_ = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration") ;

  initialSeeds_ = consumes<TrajectorySeedCollection>(conf_.getParameter<edm::InputTag>("initialSeeds")) ;
  SCEtCut_ = conf_.getParameter<double>("SCEtCut");
  fromTrackerSeeds_ = conf_.getParameter<bool>("fromTrackerSeeds") ;
  prefilteredSeeds_ = conf_.getParameter<bool>("preFilteredSeeds") ;

  auto theconsumes = consumesCollector();

  // new beamSpot tag
  beamSpotTag_ = consumes<reco::BeamSpot>(conf_.getParameter<edm::InputTag>("beamSpot")); 

  // for H/E
  applyHOverECut_ = conf_.getParameter<bool>("applyHOverECut") ;
  if (applyHOverECut_)
   {
     ElectronHcalHelper::Configuration hcalCfg ;
     hcalCfg.hOverEConeSize = conf_.getParameter<double>("hOverEConeSize") ;
     if (hcalCfg.hOverEConeSize>0)
       {
	 hcalCfg.useTowers = true ;
	 hcalCfg.hcalTowers = 
	   consumes<CaloTowerCollection>(conf_.getParameter<edm::InputTag>("hcalTowers")) ;
	 hcalCfg.hOverEPtMin = conf_.getParameter<double>("hOverEPtMin") ;
       }
     hcalHelper_ = new ElectronHcalHelper(hcalCfg) ;
     
     allowHGCal_ = conf_.getParameter<bool>("allowHGCal");
     if( allowHGCal_ ) {
       const edm::ParameterSet& hgcCfg = conf_.getParameterSet("HGCalConfig");
       hgcClusterTools_.reset( new hgcal::ClusterTools(hgcCfg, theconsumes) );
     }
     
     maxHOverEBarrel_=conf_.getParameter<double>("maxHOverEBarrel") ;
     maxHOverEEndcaps_=conf_.getParameter<double>("maxHOverEEndcaps") ;
     maxHBarrel_=conf_.getParameter<double>("maxHBarrel") ;
     maxHEndcaps_=conf_.getParameter<double>("maxHEndcaps") ;
   }

  applySigmaIEtaIEtaCut_ = conf_.getParameter<bool>("applySigmaIEtaIEtaCut");

  // apply sigma_ieta_ieta cut
  if (applySigmaIEtaIEtaCut_ == true)
    {
      maxSigmaIEtaIEtaBarrel_ = conf_.getParameter<double>("maxSigmaIEtaIEtaBarrel");
      maxSigmaIEtaIEtaEndcaps_ = conf_.getParameter<double>("maxSigmaIEtaIEtaEndcaps");
    }

  edm::ParameterSet rpset = conf_.getParameter<edm::ParameterSet>("RegionPSet");
  filterVtxTag_ = consumes<std::vector<reco::Vertex> >(rpset.getParameter<edm::InputTag> ("VertexProducer"));

  ElectronSeedGenerator::Tokens esg_tokens;
  esg_tokens.token_bs = beamSpotTag_;
  esg_tokens.token_vtx = mayConsume<reco::VertexCollection>(conf_.getParameter<edm::InputTag>("vertices"));
  esg_tokens.token_measTrkEvt= consumes<MeasurementTrackerEvent>(conf_.getParameter<edm::InputTag>("measurementTrackerEvent"));

  matcher_ = new ElectronSeedGenerator(conf_,esg_tokens) ;

  //  get collections from config
  if (applySigmaIEtaIEtaCut_ == true) {
    ebRecHitCollection_ = consumes<EcalRecHitCollection> (iConfig.getParameter<edm::InputTag>("ebRecHitCollection"));
    eeRecHitCollection_ = consumes<EcalRecHitCollection> (iConfig.getParameter<edm::InputTag>("eeRecHitCollection"));
  }

  superClusters_[0]=
    consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("barrelSuperClusters")) ;
  superClusters_[1]=
    consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("endcapSuperClusters")) ;

  // Construction of SeedFilter was in beginRun() with the comment
  // below, but it has to be done here because of ConsumesCollector
  //
  // FIXME: because of a bug presumably in tracker seeding,
  // perhaps in CombinedHitPairGenerator, badly caching some EventSetup product,
  // we must redo the SeedFilter for each run.
  if (prefilteredSeeds_) {
    SeedFilter::Tokens sf_tokens;
    sf_tokens.token_bs  = beamSpotTag_;
    sf_tokens.token_vtx = filterVtxTag_;
    edm::ConsumesCollector iC = consumesCollector();
    seedFilter_.reset(new SeedFilter(conf_, sf_tokens, iC));
  }

  //register your products
  produces<ElectronSeedCollection>() ;
}

void ElectronSeedProducer::beginRun(edm::Run const&, edm::EventSetup const&) 
{}

void ElectronSeedProducer::endRun(edm::Run const&, edm::EventSetup const&)
{}

ElectronSeedProducer::~ElectronSeedProducer()
 {
  delete hcalHelper_ ;
  delete matcher_ ;
 }

void ElectronSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
 {
  LogDebug("ElectronSeedProducer") <<"[ElectronSeedProducer::produce] entering " ;

  edm::Handle<reco::BeamSpot> theBeamSpot ;
  e.getByToken(beamSpotTag_,theBeamSpot) ;

  if (hcalHelper_)
   {
    hcalHelper_->checkSetup(iSetup) ;
    hcalHelper_->readEvent(e) ;
    if( allowHGCal_ ) {
      hgcClusterTools_->getEventSetup(iSetup);
      hgcClusterTools_->getEvent(e);
    }
   }

  // get calo geometry
  if (caloGeomCacheId_!=iSetup.get<CaloGeometryRecord>().cacheIdentifier()) {
    iSetup.get<CaloGeometryRecord>().get(caloGeom_);
    caloGeomCacheId_=iSetup.get<CaloGeometryRecord>().cacheIdentifier();
  }
  if (caloTopoCacheId_!=iSetup.get<CaloTopologyRecord>().cacheIdentifier()){
    caloTopoCacheId_=iSetup.get<CaloTopologyRecord>().cacheIdentifier();
    iSetup.get<CaloTopologyRecord>().get(caloTopo_);
  }

  matcher_->setupES(iSetup);

  // get initial TrajectorySeeds if necessary
  if (fromTrackerSeeds_)
   {
    if (!prefilteredSeeds_)
     {
      edm::Handle<TrajectorySeedCollection> hSeeds;
      e.getByToken(initialSeeds_, hSeeds);
      theInitialSeedColl = const_cast<TrajectorySeedCollection *> (hSeeds.product());
     }
    else
     { theInitialSeedColl = new TrajectorySeedCollection ; }
   }
  else
   { theInitialSeedColl = nullptr ; } // not needed in this case

  ElectronSeedCollection * seeds = new ElectronSeedCollection ;

  // loop over barrel + endcap
  for (unsigned int i=0; i<2; i++) {
    edm::Handle<SuperClusterCollection> clusters ;
    e.getByToken(superClusters_[i],clusters);
    SuperClusterRefVector clusterRefs ;
    std::vector<float> hoe1s, hoe2s ;
    filterClusters(*theBeamSpot,clusters,/*mhbhe_,*/clusterRefs,hoe1s,hoe2s,e, iSetup);
    if ((fromTrackerSeeds_) && (prefilteredSeeds_))
      { filterSeeds(e,iSetup,clusterRefs) ; }
    matcher_->run(e,iSetup,clusterRefs,hoe1s,hoe2s,theInitialSeedColl,*seeds);
  }

  // store the accumulated result
  std::unique_ptr<ElectronSeedCollection> pSeeds(seeds);
  ElectronSeedCollection::iterator is ;
  for ( is=pSeeds->begin() ; is!=pSeeds->end() ; is++ ) {
    edm::RefToBase<CaloCluster> caloCluster = is->caloCluster() ;
    SuperClusterRef superCluster = caloCluster.castTo<SuperClusterRef>() ;
    LogDebug("ElectronSeedProducer")
      << "new seed with "
      << (*is).nHits() << " hits"
      << ", charge " << (*is).getCharge()
      << " and cluster energy " << superCluster->energy()
      << " PID "<<superCluster.id() ;
  }
  e.put(std::move(pSeeds));
  if (fromTrackerSeeds_ && prefilteredSeeds_) delete theInitialSeedColl;
 }


//===============================
// Filter the superclusters
// - with EtCut
// - with HoE using calo cone
//===============================

void ElectronSeedProducer::filterClusters
 ( const reco::BeamSpot & bs,
   const edm::Handle<reco::SuperClusterCollection> & superClusters,
   SuperClusterRefVector & sclRefs,
   std::vector<float> & hoe1s, std::vector<float> & hoe2s,
   edm::Event & event, const edm::EventSetup & setup)
 {

   std::vector<float> sigmaIEtaIEtaEB_;
   std::vector<float> sigmaIEtaIEtaEE_;

  for (unsigned int i=0;i<superClusters->size();++i)
   {
    const SuperCluster & scl = (*superClusters)[i] ;
    double sclEta = EleRelPoint(scl.position(),bs.position()).eta() ;
    if (scl.energy()/cosh(sclEta)>SCEtCut_)
     {
//      if ((applyHOverECut_==true)&&((hcalHelper_->hcalESum(scl)/scl.energy()) > maxHOverE_))
//       { continue ; }
//      sclRefs.push_back(edm::Ref<reco::SuperClusterCollection>(superClusters,i)) ;
       double had1, had2, had, scle ;

       bool HoeVeto = false ;
       if (applyHOverECut_==true)
        {
         had1 = hcalHelper_->hcalESumDepth1(scl);
         had2 = hcalHelper_->hcalESumDepth2(scl);
         had = had1+had2 ;
         scle = scl.energy() ;
         int det_group = scl.seed()->hitsAndFractions()[0].first.det() ;
         int detector = scl.seed()->hitsAndFractions()[0].first.subdetId() ;
         if ( detector==EcalBarrel && (had<maxHBarrel_ || had/scle<maxHOverEBarrel_)) HoeVeto=true;
         else if( !allowHGCal_ && detector==EcalEndcap && (had<maxHEndcaps_ || had/scle<maxHOverEEndcaps_) ) HoeVeto=true;
         else if( allowHGCal_ && (detector==HcalEndcap || det_group == DetId::Forward) ) {
           float had_fraction = hgcClusterTools_->getClusterHadronFraction(*(scl.seed()));
           had1 = had_fraction*scl.seed()->energy();
           had2 = 0.;
           HoeVeto= ( had_fraction >= 0.f && had_fraction < maxHOverEEndcaps_ );
         }
         if (HoeVeto)
          {
           sclRefs.push_back(edm::Ref<reco::SuperClusterCollection>(superClusters,i)) ;
           hoe1s.push_back(had1/scle) ;
           hoe2s.push_back(had2/scle) ;
          }
        }
       else
        {
         sclRefs.push_back(edm::Ref<reco::SuperClusterCollection>(superClusters,i)) ;
         hoe1s.push_back(std::numeric_limits<float>::infinity()) ;
         hoe2s.push_back(std::numeric_limits<float>::infinity()) ;
        }
     }

    if (applySigmaIEtaIEtaCut_ == true)
      {
	noZS::EcalClusterLazyTools lazyTool_noZS(event, setup, ebRecHitCollection_, eeRecHitCollection_);
	std::vector<float> vCov = lazyTool_noZS.localCovariances(*(scl.seed()));
	int detector = scl.seed()->hitsAndFractions()[0].first.subdetId() ;
	if (detector==EcalBarrel) sigmaIEtaIEtaEB_ .push_back(edm::isNotFinite(vCov[0]) ? 0. : sqrt(vCov[0]));
	if (detector==EcalEndcap) sigmaIEtaIEtaEE_ .push_back(edm::isNotFinite(vCov[0]) ? 0. : sqrt(vCov[0]));
      }
   }
  LogDebug("ElectronSeedProducer")<<"Filtered out "<<sclRefs.size()<<" superclusters from "<<superClusters->size() ;
 }

void ElectronSeedProducer::filterSeeds
 ( edm::Event & event, const edm::EventSetup & setup,
   reco::SuperClusterRefVector & sclRefs )
 {
  for ( unsigned int i=0 ; i<sclRefs.size() ; ++i )
   {
    seedFilter_->seeds(event,setup,sclRefs[i],theInitialSeedColl) ;
    LogDebug("ElectronSeedProducer")<<"Number of Seeds: "<<theInitialSeedColl->size() ;
   }
 }

void
ElectronSeedProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("endcapSuperClusters",edm::InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALEndcapWithPreshower"));
  {
    edm::ParameterSetDescription psd0, psd1, psd2, psd3, psd4;
    psd1.add<unsigned int>("maxElement", 0);
    psd1.add<std::string>("ComponentName", std::string("StandardHitPairGenerator"));
    psd1.addUntracked<int>("useOnDemandTracker", 0);
    psd1.add<edm::InputTag>("SeedingLayers", edm::InputTag("hltMixedLayerPairs"));
    psd0.add<edm::ParameterSetDescription>("OrderedHitsFactoryPSet", psd1);

    psd2.add<double>("deltaPhiRegion", 0.4);
    psd2.add<double>("originHalfLength", 15.0);
    psd2.add<bool>("useZInVertex", true);
    psd2.add<double>("deltaEtaRegion", 0.1);
    psd2.add<double>("ptMin", 1.5 );
    psd2.add<double>("originRadius", 0.2);
    psd2.add<edm::InputTag>("VertexProducer", edm::InputTag("dummyVertices"));
    psd0.add<edm::ParameterSetDescription>("RegionPSet", psd2);
     
    psd0.add<double>("PhiMax2B",0.002);
    psd0.add<double>("hOverEPtMin",0.0);
    psd0.add<double>("PhiMax2F",0.003);
    psd0.add<bool>("searchInTIDTEC",true);
    psd0.add<double>("pPhiMax1",0.125);
    psd0.add<double>("HighPtThreshold",35.0);
    psd0.add<double>("r2MinF",-0.15);
    psd0.add<double>("maxHBarrel",0.0);
    psd0.add<double>("DeltaPhi1Low",0.23);
    psd0.add<double>("DeltaPhi1High",0.08);
    psd0.add<double>("ePhiMin1",-0.125);
    psd0.add<edm::InputTag>("hcalTowers",edm::InputTag("towerMaker"));
    psd0.add<double>("LowPtThreshold",5.0);
    psd0.add<double>("maxHOverEBarrel",0.15);
    psd0.add<double>("maxSigmaIEtaIEtaBarrel", 0.5);
    psd0.add<double>("maxSigmaIEtaIEtaEndcaps", 0.5);
    psd0.add<bool>("dynamicPhiRoad",true);
    psd0.add<double>("ePhiMax1",0.075);
    psd0.add<std::string>("measurementTrackerName","");
    psd0.add<double>("SizeWindowENeg",0.675);
    psd0.add<double>("nSigmasDeltaZ1",5.0);
    psd0.add<double>("rMaxI",0.2);
    psd0.add<double>("maxHEndcaps",0.0);
    psd0.add<bool>("preFilteredSeeds",false);
    psd0.add<double>("r2MaxF",0.15);
    psd0.add<double>("hOverEConeSize",0.15);
    psd0.add<double>("pPhiMin1",-0.075);
    psd0.add<edm::InputTag>("initialSeeds",edm::InputTag("newCombinedSeeds"));
    psd0.add<double>("deltaZ1WithVertex",25.0);
    psd0.add<double>("SCEtCut",0.0);
    psd0.add<double>("z2MaxB",0.09);
    psd0.add<bool>("fromTrackerSeeds",true);
    psd0.add<edm::InputTag>("hcalRecHits",edm::InputTag("hbhereco"));
    psd0.add<double>("z2MinB",-0.09);
    psd0.add<double>("rMinI",-0.2);
    psd0.add<double>("maxHOverEEndcaps",0.15);
    psd0.add<double>("hOverEHBMinE",0.7);
    psd0.add<bool>("useRecoVertex",false);
    psd0.add<edm::InputTag>("beamSpot",edm::InputTag("offlineBeamSpot"));
    psd0.add<edm::InputTag>("measurementTrackerEvent",edm::InputTag("MeasurementTrackerEvent"));
    psd0.add<edm::InputTag>("vertices",edm::InputTag("offlinePrimaryVerticesWithBS"));
    psd0.add<bool>("applyHOverECut",true);
    psd0.add<edm::InputTag>("ebRecHitCollection", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
    psd0.add<edm::InputTag>("eeRecHitCollection", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
    psd0.add<bool>("applySigmaIEtaIEtaCut", false);
    psd0.add<double>("DeltaPhi2F",0.012);
    psd0.add<double>("PhiMin2F",-0.003);
    psd0.add<double>("hOverEHFMinE",0.8);
    psd0.add<double>("DeltaPhi2B",0.008);
    psd0.add<double>("PhiMin2B",-0.002);
    psd0.add<bool>("allowHGCal",false);
    
    psd3.add<std::string>("ComponentName",std::string("SeedFromConsecutiveHitsCreator"));
    psd3.add<std::string>("propagator",std::string("PropagatorWithMaterial"));
    psd3.add<double>("SeedMomentumForBOFF",5.0);
    psd3.add<double>("OriginTransverseErrorMultiplier",1.0);
    psd3.add<double>("MinOneOverPtError",1.0);
    psd3.add<std::string>("magneticField",std::string(""));
    psd3.add<std::string>("TTRHBuilder",std::string("WithTrackAngle"));
    psd3.add<bool>("forceKinematicWithRegionDirection",false);

    psd4.add<edm::InputTag>("HGCEEInput",edm::InputTag("HGCalRecHit","HGCEERecHits"));
    psd4.add<edm::InputTag>("HGCFHInput",edm::InputTag("HGCalRecHit","HGCHEFRecHits"));
    psd4.add<edm::InputTag>("HGCBHInput",edm::InputTag("HGCalRecHit","HGCHEBRecHits"));
    
    psd0.add<edm::ParameterSetDescription>("HGCalConfig",psd4);

    psd0.add<edm::ParameterSetDescription>("SeedCreatorPSet",psd3);

    desc.add<edm::ParameterSetDescription>("SeedConfiguration",psd0);
  }
  desc.add<edm::InputTag>("barrelSuperClusters",edm::InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"));
  descriptions.add("ecalDrivenElectronSeeds",desc);
}
