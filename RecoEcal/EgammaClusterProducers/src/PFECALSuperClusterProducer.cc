#include "RecoEcal/EgammaClusterProducers/interface/PFECALSuperClusterProducer.h"

#include <memory>

#include "RecoEcal/EgammaClusterAlgos/interface/PFECALSuperClusterAlgo.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"

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

  use_regression = iConfig.getParameter<bool>("useRegression"); 
  eb_reg_key = iConfig.getParameter<std::string>("regressionKeyEB");
  ee_reg_key = iConfig.getParameter<std::string>("regressionKeyEE");
  gbr_record = NULL;

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


  
  inputTagPFClusters_ = 
    consumes<edm::View<reco::PFCluster> >(iConfig.getParameter<InputTag>("PFClusters"));
  inputTagPFClustersES_ = 
    consumes<reco::PFCluster::EEtoPSAssociation>(iConfig.getParameter<InputTag>("ESAssociation"));

  PFBasicClusterCollectionBarrel_ = iConfig.getParameter<string>("PFBasicClusterCollectionBarrel");
  PFSuperClusterCollectionBarrel_ = iConfig.getParameter<string>("PFSuperClusterCollectionBarrel");

  PFBasicClusterCollectionEndcap_ = iConfig.getParameter<string>("PFBasicClusterCollectionEndcap");
  PFSuperClusterCollectionEndcap_ = iConfig.getParameter<string>("PFSuperClusterCollectionEndcap");

  PFBasicClusterCollectionPreshower_ = iConfig.getParameter<string>("PFBasicClusterCollectionPreshower");
  PFSuperClusterCollectionEndcapWithPreshower_ = iConfig.getParameter<string>("PFSuperClusterCollectionEndcapWithPreshower");

  produces<reco::SuperClusterCollection>(PFSuperClusterCollectionBarrel_);  
  produces<reco::SuperClusterCollection>(PFSuperClusterCollectionEndcapWithPreshower_);   
}



PFECALSuperClusterProducer::~PFECALSuperClusterProducer() {}

void PFECALSuperClusterProducer::
beginRun(const edm::Run& iR, const edm::EventSetup& iE) {
  if(!use_regression) return;
  const GBRWrapperRcd& from_es = iE.get<GBRWrapperRcd>();
  if( !gbr_record || 
      from_es.cacheIdentifier() != gbr_record->cacheIdentifier() ) {
    gbr_record = &from_es;
    gbr_record->get(eb_reg_key.c_str(),eb_reg);    
    gbr_record->get(ee_reg_key.c_str(),ee_reg);
  }  
}


void PFECALSuperClusterProducer::produce(edm::Event& iEvent, 
				const edm::EventSetup& iSetup) {
  

  //Load the pfcluster collections
  edm::Handle<edm::View<reco::PFCluster> > pfclustersHandle;
  iEvent.getByToken( inputTagPFClusters_, pfclustersHandle );  

  edm::Handle<reco::PFCluster::EEtoPSAssociation > psAssociationHandle;
  iEvent.getByToken( inputTagPFClustersES_,  psAssociationHandle);


  // do clustering
  superClusterAlgo_.loadAndSortPFClusters(*pfclustersHandle,
					  *psAssociationHandle);
  superClusterAlgo_.run();

  //store in the event
  iEvent.put(superClusterAlgo_.getEBOutputSCCollection(),
	     PFSuperClusterCollectionBarrel_);
  iEvent.put(superClusterAlgo_.getEEOutputSCCollection(), 
	     PFSuperClusterCollectionEndcapWithPreshower_);
}

double PFECALSuperClusterProducer::
calculateRegressedEnergy(const reco::SuperCluster& sc) {
  edm::Ptr<reco::PFCluster> seed(sc.seed());
  memset(rinputs,0,33*sizeof(float));
  switch( seed->layer() ) {
  case PFLayer::ECAL_BARREL:
    /*
      nVtx
      scEta
      scPhi
      scEtaWidth
      scPhiWidth
      scSeedR9
      scSeedRawEnergy/scRawEnergy
      scSeedEmax/scRawEnergy
      scSeedE2nd/scRawEnergy
      scSeedLeftRightAsym
      scSeedTopBottomAsym
      scSeedSigmaIetaIeta
      scSeedSigmaIetaIphi
      scSeedSigmaIphiIphi
      N_ECALClusters
      clusterMaxDR
      clusterMaxDRDPhi
      clusterMaxDRDEta
      clusterMaxDRRawEnergy/scRawEnergy
      clusterRawEnergy[0]/scRawEnergy
      clusterRawEnergy[1]/scRawEnergy
      clusterRawEnergy[2]/scRawEnergy
      clusterDPhiToSeed[0]
      clusterDPhiToSeed[1]
      clusterDPhiToSeed[2]
      clusterDEtaToSeed[0]
      clusterDEtaToSeed[1]
      clusterDEtaToSeed[2]
      scSeedCryEta
      scSeedCryPhi
      scSeedCryIeta
      scSeedCryIphi
      scCalibratedEnergy
    */
    return eb_reg->GetResponse(rinputs);
    break;
  case PFLayer::ECAL_ENDCAP:
    break;
    /*
      nVtx
      scEta
      scPhi
      scEtaWidth
      scPhiWidth
      scSeedR9
      scSeedRawEnergy/scRawEnergy
      scSeedEmax/scRawEnergy
      scSeedE2nd/scRawEnergy
      scSeedLeftRightAsym
      scSeedTopBottomAsym
      scSeedSigmaIetaIeta
      scSeedSigmaIetaIphi
      scSeedSigmaIphiIphi
      N_ECALClusters
      clusterMaxDR
      clusterMaxDRDPhi
      clusterMaxDRDEta
      clusterMaxDRRawEnergy/scRawEnergy
      clusterRawEnergy[0]/scRawEnergy
      clusterRawEnergy[1]/scRawEnergy
      clusterRawEnergy[2]/scRawEnergy
      clusterDPhiToSeed[0]
      clusterDPhiToSeed[1]
      clusterDPhiToSeed[2]
      clusterDEtaToSeed[0]
      clusterDEtaToSeed[1]
      clusterDEtaToSeed[2]
      scPreshowerEnergy/scRawEnergy
      scCalibratedEnergy
    */
    return ee_reg->GetResponse(rinputs);
  default:
   throw cms::Exception("PFECALSuperClusterProducer::calculateRegressedEnergy")
     << "Supercluster seed is either EB nor EE!" << std::endl;
  }
  return -1;
}


