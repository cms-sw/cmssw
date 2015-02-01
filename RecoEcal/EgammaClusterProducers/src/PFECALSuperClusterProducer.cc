#include "RecoEcal/EgammaClusterProducers/interface/PFECALSuperClusterProducer.h"

#include <memory>

#include "RecoEcal/EgammaClusterAlgos/interface/PFECALSuperClusterAlgo.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "TVector2.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace std;
using namespace edm;

namespace {
  const std::string ClusterType__BOX("Box");
  const std::string ClusterType__Mustache("Mustache");

  const std::string EnergyWeight__Raw("Raw");
  const std::string EnergyWeight__CalibratedNoPS("CalibratedNoPS");
  const std::string EnergyWeight__CalibratedTotal("CalibratedTotal");
}

PFECALSuperClusterProducer::PFECALSuperClusterProducer(const edm::ParameterSet& iConfig)
{
    
  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  superClusterAlgo_.setUseRegression(iConfig.getParameter<bool>("useRegression")); 
    
  superClusterAlgo_.setTokens(iConfig,consumesCollector());
   
  std::string _typename = iConfig.getParameter<std::string>("ClusteringType");
  if( _typename == ClusterType__BOX ) {
    _theclusteringtype = PFECALSuperClusterAlgo::kBOX;
  } else if ( _typename == ClusterType__Mustache ) {
    _theclusteringtype = PFECALSuperClusterAlgo::kMustache;
  } else {
    throw cms::Exception("InvalidClusteringType") 
      << "You have not chosen a valid clustering type," 
      << " please choose from \"Box\" or \"Mustache\"!";
  }

  std::string _weightname = iConfig.getParameter<std::string>("EnergyWeight");
  if( _weightname == EnergyWeight__Raw ) {
    _theenergyweight = PFECALSuperClusterAlgo::kRaw;
  } else if ( _weightname == EnergyWeight__CalibratedNoPS ) {
    _theenergyweight = PFECALSuperClusterAlgo::kCalibratedNoPS;
  } else if ( _weightname == EnergyWeight__CalibratedTotal) {
    _theenergyweight = PFECALSuperClusterAlgo::kCalibratedTotal;
  } else {
    throw cms::Exception("InvalidClusteringType") 
      << "You have not chosen a valid energy weighting scheme," 
      << " please choose from \"Raw\", \"CalibratedNoPS\", or"
      << " \"CalibratedTotal\"!";
  }
  

  // parameters for clustering
  bool seedThresholdIsET = iConfig.getParameter<bool>("seedThresholdIsET");

  bool useDynamicDPhi = iConfig.getParameter<bool>("useDynamicDPhiWindow");

  double threshPFClusterSeedBarrel = iConfig.getParameter<double>("thresh_PFClusterSeedBarrel");
  double threshPFClusterBarrel = iConfig.getParameter<double>("thresh_PFClusterBarrel");

  double threshPFClusterSeedEndcap = iConfig.getParameter<double>("thresh_PFClusterSeedEndcap");
  double threshPFClusterEndcap = iConfig.getParameter<double>("thresh_PFClusterEndcap");
  
  double phiwidthSuperClusterBarrel = iConfig.getParameter<double>("phiwidth_SuperClusterBarrel");
  double etawidthSuperClusterBarrel = iConfig.getParameter<double>("etawidth_SuperClusterBarrel");

  double phiwidthSuperClusterEndcap = iConfig.getParameter<double>("phiwidth_SuperClusterEndcap");
  double etawidthSuperClusterEndcap = iConfig.getParameter<double>("etawidth_SuperClusterEndcap");

  double threshPFClusterES = iConfig.getParameter<double>("thresh_PFClusterES");

  //double threshPFClusterMustacheOutBarrel = iConfig.getParameter<double>("thresh_PFClusterMustacheOutBarrel");
  //double threshPFClusterMustacheOutEndcap = iConfig.getParameter<double>("thresh_PFClusterMustacheOutEndcap");

  double doSatelliteClusterMerge = 
    iConfig.getParameter<bool>("doSatelliteClusterMerge");
  double satelliteClusterSeedThreshold = 
    iConfig.getParameter<double>("satelliteClusterSeedThreshold");
  double satelliteMajorityFraction = 
    iConfig.getParameter<double>("satelliteMajorityFraction");

  superClusterAlgo_.setVerbosityLevel(verbose_);
  superClusterAlgo_.setClusteringType(_theclusteringtype);
  superClusterAlgo_.setEnergyWeighting(_theenergyweight);
  superClusterAlgo_.setUseETForSeeding(seedThresholdIsET);
  superClusterAlgo_.setUseDynamicDPhi(useDynamicDPhi);

  superClusterAlgo_.setThreshSuperClusterEt( iConfig.getParameter<double>("thresh_SCEt") );
  
  superClusterAlgo_.setThreshPFClusterSeedBarrel( threshPFClusterSeedBarrel );
  superClusterAlgo_.setThreshPFClusterBarrel( threshPFClusterBarrel );

  superClusterAlgo_.setThreshPFClusterSeedEndcap( threshPFClusterSeedEndcap );
  superClusterAlgo_.setThreshPFClusterEndcap( threshPFClusterEndcap );

  superClusterAlgo_.setPhiwidthSuperClusterBarrel( phiwidthSuperClusterBarrel );
  superClusterAlgo_.setEtawidthSuperClusterBarrel( etawidthSuperClusterBarrel );

  superClusterAlgo_.setPhiwidthSuperClusterEndcap( phiwidthSuperClusterEndcap );
  superClusterAlgo_.setEtawidthSuperClusterEndcap( etawidthSuperClusterEndcap );

  superClusterAlgo_.setThreshPFClusterES( threshPFClusterES );

  superClusterAlgo_.setSatelliteMerging( doSatelliteClusterMerge );
  superClusterAlgo_.setSatelliteThreshold( satelliteClusterSeedThreshold );
  superClusterAlgo_.setMajorityFraction( satelliteMajorityFraction );
  //superClusterAlgo_.setThreshPFClusterMustacheOutBarrel( threshPFClusterMustacheOutBarrel );
  //superClusterAlgo_.setThreshPFClusterMustacheOutEndcap( threshPFClusterMustacheOutEndcap );

  //Load the ECAL energy calibration
  thePFEnergyCalibration_ = 
    std::shared_ptr<PFEnergyCalibration>(new PFEnergyCalibration());
  superClusterAlgo_.setPFClusterCalibration(thePFEnergyCalibration_);
  superClusterAlgo_.setUsePS(iConfig.getParameter<bool>("use_preshower"));

  bool applyCrackCorrections_ = iConfig.getParameter<bool>("applyCrackCorrections");
  superClusterAlgo_.setCrackCorrections(applyCrackCorrections_);
  
  PFBasicClusterCollectionBarrel_ = iConfig.getParameter<string>("PFBasicClusterCollectionBarrel");
  PFSuperClusterCollectionBarrel_ = iConfig.getParameter<string>("PFSuperClusterCollectionBarrel");

  PFBasicClusterCollectionEndcap_ = iConfig.getParameter<string>("PFBasicClusterCollectionEndcap");
  PFSuperClusterCollectionEndcap_ = iConfig.getParameter<string>("PFSuperClusterCollectionEndcap");

  PFBasicClusterCollectionPreshower_ = iConfig.getParameter<string>("PFBasicClusterCollectionPreshower");
  PFSuperClusterCollectionEndcapWithPreshower_ = iConfig.getParameter<string>("PFSuperClusterCollectionEndcapWithPreshower");

  PFClusterAssociationEBEE_ = "PFClusterAssociationEBEE";
  PFClusterAssociationES_ = "PFClusterAssociationES";
  
  produces<reco::SuperClusterCollection>(PFSuperClusterCollectionBarrel_);  
  produces<reco::SuperClusterCollection>(PFSuperClusterCollectionEndcapWithPreshower_);
  produces<reco::CaloClusterCollection>(PFBasicClusterCollectionBarrel_);
  produces<reco::CaloClusterCollection>(PFBasicClusterCollectionEndcap_);
  produces<reco::CaloClusterCollection>(PFBasicClusterCollectionPreshower_);
  produces<edm::ValueMap<reco::CaloClusterPtr> >(PFClusterAssociationEBEE_);
  produces<edm::ValueMap<reco::CaloClusterPtr> >(PFClusterAssociationES_);
}



PFECALSuperClusterProducer::~PFECALSuperClusterProducer() {}

void PFECALSuperClusterProducer::
beginLuminosityBlock(LuminosityBlock const& iL, EventSetup const& iE) {
  superClusterAlgo_.update(iE); 
}


void PFECALSuperClusterProducer::produce(edm::Event& iEvent, 
				const edm::EventSetup& iSetup) {
  
  // do clustering
  superClusterAlgo_.loadAndSortPFClusters(iEvent);
  superClusterAlgo_.run();
  
  //build collections of output CaloClusters from the used PFClusters
  std::auto_ptr<reco::CaloClusterCollection> caloClustersEB(new reco::CaloClusterCollection);
  std::auto_ptr<reco::CaloClusterCollection> caloClustersEE(new reco::CaloClusterCollection);
  std::auto_ptr<reco::CaloClusterCollection> caloClustersES(new reco::CaloClusterCollection);
  
  std::map<reco::CaloClusterPtr, unsigned int> pfClusterMapEB; //maps of pfclusters to caloclusters 
  std::map<reco::CaloClusterPtr, unsigned int> pfClusterMapEE;
  std::map<reco::CaloClusterPtr, unsigned int> pfClusterMapES;
  
  //fill calocluster collections and maps
  for( const auto& ebsc : *(superClusterAlgo_.getEBOutputSCCollection()) ) {
    for (reco::CaloCluster_iterator pfclus = ebsc.clustersBegin(); pfclus!=ebsc.clustersEnd(); ++pfclus) {
      if (!pfClusterMapEB.count(*pfclus)) {
        reco::CaloCluster caloclus(**pfclus);
        caloClustersEB->push_back(caloclus);
        pfClusterMapEB[*pfclus] = caloClustersEB->size() - 1;
      }
      else {
        throw cms::Exception("PFECALSuperClusterProducer::produce")
            << "Found an EB pfcluster matched to more than one EB supercluster!" 
            << std::dec << std::endl;
      }
    }
  }
  for( const auto& eesc : *(superClusterAlgo_.getEEOutputSCCollection()) ) {
    for (reco::CaloCluster_iterator pfclus = eesc.clustersBegin(); pfclus!=eesc.clustersEnd(); ++pfclus) {
      if (!pfClusterMapEE.count(*pfclus)) {
        reco::CaloCluster caloclus(**pfclus);
        caloClustersEE->push_back(caloclus);
        pfClusterMapEE[*pfclus] = caloClustersEE->size() - 1;
      }
      else {
        throw cms::Exception("PFECALSuperClusterProducer::produce")
            << "Found an EE pfcluster matched to more than one EE supercluster!" 
            << std::dec << std::endl;
      }
    }
    for (reco::CaloCluster_iterator pfclus = eesc.preshowerClustersBegin(); pfclus!=eesc.preshowerClustersEnd(); ++pfclus) {
      if (!pfClusterMapES.count(*pfclus)) {
        reco::CaloCluster caloclus(**pfclus);
        caloClustersES->push_back(caloclus);
        pfClusterMapES[*pfclus] = caloClustersES->size() - 1;
      }
      else {
        throw cms::Exception("PFECALSuperClusterProducer::produce")
            << "Found an ES pfcluster matched to more than one EE supercluster!" 
            << std::dec << std::endl;
      }
    }
  }
  
  //create ValueMaps from output CaloClusters back to original PFClusters
  std::auto_ptr<edm::ValueMap<reco::CaloClusterPtr> > pfClusterAssociationEBEE(new edm::ValueMap<reco::CaloClusterPtr>);
  std::auto_ptr<edm::ValueMap<reco::CaloClusterPtr> > pfClusterAssociationES(new edm::ValueMap<reco::CaloClusterPtr>);    
  
  //vectors to fill ValueMaps
  std::vector<reco::CaloClusterPtr> clusptrsEB(caloClustersEB->size());
  std::vector<reco::CaloClusterPtr> clusptrsEE(caloClustersEE->size());
  std::vector<reco::CaloClusterPtr> clusptrsES(caloClustersES->size());  
  
  //put calocluster output collections in event and get orphan handles to create ptrs
  const edm::OrphanHandle<reco::CaloClusterCollection> &caloClusHandleEB = iEvent.put(caloClustersEB,PFBasicClusterCollectionBarrel_);
  const edm::OrphanHandle<reco::CaloClusterCollection> &caloClusHandleEE = iEvent.put(caloClustersEE,PFBasicClusterCollectionEndcap_);
  const edm::OrphanHandle<reco::CaloClusterCollection> &caloClusHandleES = iEvent.put(caloClustersES,PFBasicClusterCollectionPreshower_);
    
  //relink superclusters to output caloclusters and fill vectors for ValueMaps
  for( auto& ebsc : *(superClusterAlgo_.getEBOutputSCCollection()) ) {
    reco::CaloClusterPtr seedptr(caloClusHandleEB,pfClusterMapEB[ebsc.seed()]);
    ebsc.setSeed(seedptr);
    
    reco::CaloClusterPtrVector clusters;
    for (reco::CaloCluster_iterator pfclus = ebsc.clustersBegin(); pfclus!=ebsc.clustersEnd(); ++pfclus) {
      int caloclusidx = pfClusterMapEB[*pfclus];
      reco::CaloClusterPtr clusptr(caloClusHandleEB,caloclusidx);
      clusters.push_back(clusptr);
      clusptrsEB[caloclusidx] = *pfclus;
    }
    ebsc.setClusters(clusters);
  }
  for( auto& eesc : *(superClusterAlgo_.getEEOutputSCCollection()) ) {
    reco::CaloClusterPtr seedptr(caloClusHandleEE,pfClusterMapEE[eesc.seed()]);
    eesc.setSeed(seedptr);
    
    reco::CaloClusterPtrVector clusters;
    for (reco::CaloCluster_iterator pfclus = eesc.clustersBegin(); pfclus!=eesc.clustersEnd(); ++pfclus) {
      int caloclusidx = pfClusterMapEE[*pfclus];
      reco::CaloClusterPtr clusptr(caloClusHandleEE,caloclusidx);
      clusters.push_back(clusptr);
      clusptrsEE[caloclusidx] = *pfclus;
    }
    eesc.setClusters(clusters);
    
    reco::CaloClusterPtrVector psclusters;
    for (reco::CaloCluster_iterator pfclus = eesc.preshowerClustersBegin(); pfclus!=eesc.preshowerClustersEnd(); ++pfclus) {
      int caloclusidx = pfClusterMapES[*pfclus];
      reco::CaloClusterPtr clusptr(caloClusHandleES,caloclusidx);
      psclusters.push_back(clusptr);
      clusptrsES[caloclusidx] = *pfclus;
    }
    eesc.setPreshowerClusters(psclusters);  
  }
    
  //fill association maps from output CaloClusters back to original PFClusters
  edm::ValueMap<reco::CaloClusterPtr>::Filler fillerEBEE(*pfClusterAssociationEBEE);
  fillerEBEE.insert(caloClusHandleEB, clusptrsEB.begin(), clusptrsEB.end());
  fillerEBEE.insert(caloClusHandleEE, clusptrsEE.begin(), clusptrsEE.end());
  fillerEBEE.fill();
  
  edm::ValueMap<reco::CaloClusterPtr>::Filler fillerES(*pfClusterAssociationES);
  fillerES.insert(caloClusHandleES, clusptrsES.begin(), clusptrsES.end());  
  fillerES.fill();  

  //store in the event
  iEvent.put(pfClusterAssociationEBEE,PFClusterAssociationEBEE_);
  iEvent.put(pfClusterAssociationES,PFClusterAssociationES_);
  iEvent.put(superClusterAlgo_.getEBOutputSCCollection(),
	     PFSuperClusterCollectionBarrel_);
  iEvent.put(superClusterAlgo_.getEEOutputSCCollection(), 
	     PFSuperClusterCollectionEndcapWithPreshower_);
}

void PFECALSuperClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("PFSuperClusterCollectionEndcap","particleFlowSuperClusterECALEndcap");
  desc.add<bool>("doSatelliteClusterMerge",false);
  desc.add<double>("thresh_PFClusterBarrel",0.0);
  desc.add<std::string>("PFBasicClusterCollectionBarrel","particleFlowBasicClusterECALBarrel");
  desc.add<bool>("useRegression",true);
  desc.add<double>("satelliteMajorityFraction",0.5);
  desc.add<double>("thresh_PFClusterEndcap",0.0);
  desc.add<edm::InputTag>("ESAssociation",edm::InputTag("particleFlowClusterECAL"));
  desc.add<std::string>("PFBasicClusterCollectionPreshower","particleFlowBasicClusterECALPreshower");
  desc.add<bool>("use_preshower",true);
  desc.addUntracked<bool>("verbose",false);
  desc.add<double>("thresh_SCEt",4.0);
  desc.add<double>("etawidth_SuperClusterEndcap",0.04);
  desc.add<double>("phiwidth_SuperClusterEndcap",0.6);
  desc.add<bool>("useDynamicDPhiWindow",true);
  desc.add<std::string>("PFSuperClusterCollectionBarrel","particleFlowSuperClusterECALBarrel");
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("regressionKeyEE","pfscecal_EECorrection_offline_v1");
    psd0.add<edm::InputTag>("ecalRecHitsEE",edm::InputTag("ecalRecHit","EcalRecHitsEE"));
    psd0.add<edm::InputTag>("ecalRecHitsEB",edm::InputTag("ecalRecHit","EcalRecHitsEB"));
    psd0.add<std::string>("regressionKeyEB","pfscecal_EBCorrection_offline_v1");
    psd0.add<edm::InputTag>("vertexCollection",edm::InputTag("offlinePrimaryVertices"));
    desc.add<edm::ParameterSetDescription>("regressionConfig",psd0);
  }
  desc.add<bool>("applyCrackCorrections",false);
  desc.add<double>("satelliteClusterSeedThreshold",50.0);
  desc.add<double>("etawidth_SuperClusterBarrel",0.04);
  desc.add<std::string>("PFBasicClusterCollectionEndcap","particleFlowBasicClusterECALEndcap");
  desc.add<edm::InputTag>("PFClusters",edm::InputTag("particleFlowClusterECAL"));
  desc.add<double>("thresh_PFClusterSeedBarrel",1.0);
  desc.add<std::string>("ClusteringType","Mustache");
  desc.add<std::string>("EnergyWeight","Raw");
  desc.add<edm::InputTag>("BeamSpot",edm::InputTag("offlineBeamSpot"));
  desc.add<double>("thresh_PFClusterSeedEndcap",1.0);
  desc.add<double>("phiwidth_SuperClusterBarrel",0.6);
  desc.add<double>("thresh_PFClusterES",0.0);
  desc.add<bool>("seedThresholdIsET",true);
  desc.add<std::string>("PFSuperClusterCollectionEndcapWithPreshower","particleFlowSuperClusterECALEndcapWithPreshower");
  descriptions.add("particleFlowSuperClusterECALMustache",desc);
}
