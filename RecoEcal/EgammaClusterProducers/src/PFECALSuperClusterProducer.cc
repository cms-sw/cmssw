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

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"


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

  use_regression = iConfig.getParameter<bool>("useRegression"); 
  eb_reg_key = iConfig.getParameter<std::string>("regressionKeyEB");
  ee_reg_key = iConfig.getParameter<std::string>("regressionKeyEE");
  gbr_record = NULL;
  topo_record = NULL;

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
  inputTagEBRecHits_ = 
    mayConsume<EcalRecHitCollection>(iConfig.getParameter<InputTag>("ecalRecHitsEB"));
  inputTagEERecHits_ = 
    mayConsume<EcalRecHitCollection>(iConfig.getParameter<InputTag>("ecalRecHitsEE"));
  inputTagVertices_ = 
    mayConsume<reco::VertexCollection>(iConfig.getParameter<InputTag>("vertexCollection"));


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
  const GBRWrapperRcd& gbrfrom_es = iE.get<GBRWrapperRcd>();
  if( !gbr_record || 
      gbrfrom_es.cacheIdentifier() != gbr_record->cacheIdentifier() ) {
    gbr_record = &gbrfrom_es;
    gbr_record->get(eb_reg_key.c_str(),eb_reg);    
    gbr_record->get(ee_reg_key.c_str(),ee_reg);
  }  
  const CaloTopologyRecord& topofrom_es = iE.get<CaloTopologyRecord>();
  if( !topo_record ||
      topofrom_es.cacheIdentifier() != topo_record->cacheIdentifier() ) {
    topo_record = &topofrom_es;
    topo_record->get(calotopo);
  }      
}


void PFECALSuperClusterProducer::produce(edm::Event& iEvent, 
				const edm::EventSetup& iSetup) {
  

  //Load the pfcluster collections
  edm::Handle<edm::View<reco::PFCluster> > pfclustersHandle;
  iEvent.getByToken( inputTagPFClusters_, pfclustersHandle );  

  edm::Handle<reco::PFCluster::EEtoPSAssociation > psAssociationHandle;
  iEvent.getByToken( inputTagPFClustersES_,  psAssociationHandle);

  edm::Handle<reco::VertexCollection> vertices;
  edm::Handle<EcalRecHitCollection>  rechitsEB,rechitsEE;

  // do clustering
  superClusterAlgo_.loadAndSortPFClusters(*pfclustersHandle,
					  *psAssociationHandle);
  superClusterAlgo_.run();

  if( use_regression ) {
    edm::Handle<reco::VertexCollection> vertices;
    edm::Handle<EcalRecHitCollection>  rechitsEB,rechitsEE;
    iEvent.getByToken(inputTagEBRecHits_,rechitsEB);
    iEvent.getByToken(inputTagEERecHits_,rechitsEE);
    iEvent.getByToken(inputTagVertices_,vertices);
    double cor = 0.0;
    for( auto& ebsc : *(superClusterAlgo_.getEBOutputSCCollection()) ) {
      cor = getRegressionCorrection(ebsc,
				    vertices,
				    rechitsEB,
				    rechitsEE,
				    iSetup);
      ebsc.setEnergy(cor*ebsc.energy());
    }
    for( auto& eesc : *(superClusterAlgo_.getEEOutputSCCollection()) ) {
      cor = getRegressionCorrection(eesc,
				    vertices,
				    rechitsEB,
				    rechitsEE,
				    iSetup);
      eesc.setEnergy(cor*eesc.energy());
    }
  }

  //store in the event
  iEvent.put(superClusterAlgo_.getEBOutputSCCollection(),
	     PFSuperClusterCollectionBarrel_);
  iEvent.put(superClusterAlgo_.getEEOutputSCCollection(), 
	     PFSuperClusterCollectionEndcapWithPreshower_);
}

double PFECALSuperClusterProducer::
getRegressionCorrection(const reco::SuperCluster& sc,
			const edm::Handle<reco::VertexCollection>& vertices,
			const edm::Handle<EcalRecHitCollection>& rechitsEB,
			const edm::Handle<EcalRecHitCollection>& rechitsEE,
			const edm::EventSetup& es) {  
  memset(rinputs,0,33*sizeof(float));
  const double rawEnergy = sc.rawEnergy(), calibEnergy = sc.energy();
  const edm::Ptr<reco::PFCluster> seed(sc.seed());
  const size_t nVtx = vertices->size();
  float maxDR=999., maxDRDPhi=999., maxDRDEta=999., maxDRRawEnergy=0.;
  float subClusRawE[3], subClusDPhi[3], subClusDEta[3];
  memset(subClusRawE,0,3*sizeof(float));
  memset(subClusDPhi,0,3*sizeof(float));
  memset(subClusDEta,0,3*sizeof(float));
  size_t iclus=0;
  for( auto clus = sc.clustersBegin()+1; clus != sc.clustersEnd(); ++clus ) {
    const float this_dr = reco::deltaR(**clus, *seed);
     if(this_dr > maxDR || maxDR == 999.0f) {
       maxDR = this_dr;
       maxDRDEta = (*clus)->eta() - seed->eta();
       maxDRDPhi = TVector2::Phi_mpi_pi((*clus)->phi() - seed->phi());
       maxDRRawEnergy = (*clus)->energy();
     }
     if( iclus++ < 3 ) {
       subClusRawE[iclus] = (*clus)->energy();
       subClusDEta[iclus] = (*clus)->eta() - seed->eta();
       subClusDPhi[iclus] = TVector2::Phi_mpi_pi((*clus)->phi() - seed->phi());
     }
  }
  float scPreshowerSum = 0.0;
  for( auto psclus = sc.preshowerClustersBegin(); 
       psclus != sc.preshowerClustersEnd(); ++psclus ) {
    scPreshowerSum += (*psclus)->energy();
  }
  switch( seed->layer() ) {
  case PFLayer::ECAL_BARREL:
    {
      const float eMax = EcalClusterTools::eMax( *seed, &*rechitsEB );
      const float e2nd = EcalClusterTools::e2nd( *seed, &*rechitsEB );
      const float e3x3 = EcalClusterTools::e3x3( *seed,
						 &*rechitsEB, 
						 &*calotopo  );
      const float eTop = EcalClusterTools::eTop( *seed, 
						 &*rechitsEB, 
						 &*calotopo );
      const float eBottom = EcalClusterTools::eBottom( *seed, 
						       &*rechitsEB, 
						       &*calotopo );
      const float eLeft = EcalClusterTools::eLeft( *seed, 
						   &*rechitsEB, 
						   &*calotopo );
      const float eRight = EcalClusterTools::eRight( *seed, 
						     &*rechitsEB, 
						     &*calotopo );
      const float eLpeR = eLeft + eRight;
      const float eTpeB = eTop + eBottom;
      const float eLmeR = eLeft - eRight;
      const float eTmeB = eTop - eBottom;
      std::vector<float> vCov = 
	EcalClusterTools::localCovariances( *seed, &*rechitsEB, &*calotopo );
      const float see = (isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
      const float spp = (isnan(vCov[2]) ? 0. : sqrt(vCov[2]));
      float sep = 0.;
      if (see*spp > 0)
        sep = vCov[1] / (see * spp);
      else if (vCov[1] > 0)
        sep = 1.0;
      else
        sep = -1.0;
      float cryPhi, cryEta, thetatilt, phitilt;
      int ieta, iphi;
      ecl_.localCoordsEB(*seed, es, cryEta, cryPhi, 
			 ieta, iphi, thetatilt, phitilt);
      rinputs[0] = nVtx;                          //nVtx
      rinputs[1] = sc.eta();                      //scEta
      rinputs[2] = sc.phi();                      //scPhi
      rinputs[3] = sc.etaWidth();                 //scEtaWidth
      rinputs[4] = sc.phiWidth();                 //scPhiWidth
      rinputs[5] = e3x3/rawEnergy;                //scSeedR9
      rinputs[6] = sc.seed()->energy()/rawEnergy; //scSeedRawEnergy/scRawEnergy
      rinputs[7] = eMax/rawEnergy;                //scSeedEmax/scRawEnergy
      rinputs[8] = e2nd/rawEnergy;                //scSeedE2nd/scRawEnergy
      rinputs[9] = (eLpeR!=0. ? eLmeR/eLpeR : 0.);//scSeedLeftRightAsym
      rinputs[10] = (eTpeB!=0.? eTmeB/eTpeB : 0.);//scSeedTopBottomAsym
      rinputs[11] = see;                          //scSeedSigmaIetaIeta
      rinputs[12] = sep;                          //scSeedSigmaIetaIphi
      rinputs[13] = spp;                          //scSeedSigmaIphiIphi
      rinputs[14] = sc.clustersSize()-1;          //N_ECALClusters
      rinputs[15] = maxDR;                        //clusterMaxDR
      rinputs[16] = maxDRDPhi;                    //clusterMaxDRDPhi
      rinputs[17] = maxDRDEta;                    //clusterMaxDRDEta
      rinputs[18] = maxDRRawEnergy/rawEnergy; //clusMaxDRRawEnergy/scRawEnergy
      rinputs[19] = subClusRawE[0]/rawEnergy; //clusterRawEnergy[0]/scRawEnergy
      rinputs[20] = subClusRawE[1]/rawEnergy; //clusterRawEnergy[1]/scRawEnergy
      rinputs[21] = subClusRawE[2]/rawEnergy; //clusterRawEnergy[2]/scRawEnergy
      rinputs[22] = subClusDPhi[0];               //clusterDPhiToSeed[0]
      rinputs[23] = subClusDPhi[1];               //clusterDPhiToSeed[1]
      rinputs[24] = subClusDPhi[2];               //clusterDPhiToSeed[2]
      rinputs[25] = subClusDEta[0];               //clusterDEtaToSeed[0]
      rinputs[26] = subClusDEta[1];               //clusterDEtaToSeed[1]
      rinputs[27] = subClusDEta[2];               //clusterDEtaToSeed[2]
      rinputs[28] = cryEta;                       //scSeedCryEta
      rinputs[29] = cryPhi;                       //scSeedCryPhi
      rinputs[30] = ieta;                         //scSeedCryIeta
      rinputs[31] = iphi;                         //scSeedCryIphi
      rinputs[32] = calibEnergy;                  //scCalibratedEnergy
    }
    return eb_reg->GetResponse(rinputs);
    break;
  case PFLayer::ECAL_ENDCAP:
    {
      const float eMax = EcalClusterTools::eMax( *seed, &*rechitsEE );
      const float e2nd = EcalClusterTools::e2nd( *seed, &*rechitsEE );
      const float e3x3 = EcalClusterTools::e3x3( *seed,
						 &*rechitsEE, 
						 &*calotopo  );
      const float eTop = EcalClusterTools::eTop( *seed, 
						 &*rechitsEE, 
						 &*calotopo );
      const float eBottom = EcalClusterTools::eBottom( *seed, 
						       &*rechitsEE, 
						       &*calotopo );
      const float eLeft = EcalClusterTools::eLeft( *seed, 
						   &*rechitsEE, 
						   &*calotopo );
      const float eRight = EcalClusterTools::eRight( *seed, 
						     &*rechitsEE, 
						     &*calotopo );
      const float eLpeR = eLeft + eRight;
      const float eTpeB = eTop + eBottom;
      const float eLmeR = eLeft - eRight;
      const float eTmeB = eTop - eBottom;
      std::vector<float> vCov = 
	EcalClusterTools::localCovariances( *seed, &*rechitsEE, &*calotopo );
      const float see = (isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
      const float spp = (isnan(vCov[2]) ? 0. : sqrt(vCov[2]));
      float sep = 0.;
      if (see*spp > 0)
        sep = vCov[1] / (see * spp);
      else if (vCov[1] > 0)
        sep = 1.0;
      else
        sep = -1.0;      
      rinputs[0] = nVtx;                          //nVtx
      rinputs[1] = sc.eta();                      //scEta
      rinputs[2] = sc.phi();                      //scPhi
      rinputs[3] = sc.etaWidth();                 //scEtaWidth
      rinputs[4] = sc.phiWidth();                 //scPhiWidth
      rinputs[5] = e3x3/rawEnergy;                //scSeedR9
      rinputs[6] = sc.seed()->energy()/rawEnergy; //scSeedRawEnergy/scRawEnergy
      rinputs[7] = eMax/rawEnergy;                //scSeedEmax/scRawEnergy
      rinputs[8] = e2nd/rawEnergy;                //scSeedE2nd/scRawEnergy
      rinputs[9] = (eLpeR!=0. ? eLmeR/eLpeR : 0.);//scSeedLeftRightAsym
      rinputs[10] = (eTpeB!=0.? eTmeB/eTpeB : 0.);//scSeedTopBottomAsym
      rinputs[11] = see;                          //scSeedSigmaIetaIeta
      rinputs[12] = sep;                          //scSeedSigmaIetaIphi
      rinputs[13] = spp;                          //scSeedSigmaIphiIphi
      rinputs[14] = sc.clustersSize()-1;          //N_ECALClusters
      rinputs[15] = maxDR;                        //clusterMaxDR
      rinputs[16] = maxDRDPhi;                    //clusterMaxDRDPhi
      rinputs[17] = maxDRDEta;                    //clusterMaxDRDEta
      rinputs[18] = maxDRRawEnergy/rawEnergy; //clusMaxDRRawEnergy/scRawEnergy
      rinputs[19] = subClusRawE[0]/rawEnergy; //clusterRawEnergy[0]/scRawEnergy
      rinputs[20] = subClusRawE[1]/rawEnergy; //clusterRawEnergy[1]/scRawEnergy
      rinputs[21] = subClusRawE[2]/rawEnergy; //clusterRawEnergy[2]/scRawEnergy
      rinputs[22] = subClusDPhi[0];               //clusterDPhiToSeed[0]
      rinputs[23] = subClusDPhi[1];               //clusterDPhiToSeed[1]
      rinputs[24] = subClusDPhi[2];               //clusterDPhiToSeed[2]
      rinputs[25] = subClusDEta[0];               //clusterDEtaToSeed[0]
      rinputs[26] = subClusDEta[1];               //clusterDEtaToSeed[1]
      rinputs[27] = subClusDEta[2];               //clusterDEtaToSeed[2]
      rinputs[28] = scPreshowerSum/rawEnergy;   //scPreshowerEnergy/scRawEnergy
      rinputs[29] = calibEnergy;                  //scCalibratedEnergy
    }
    return ee_reg->GetResponse(rinputs);
    break;    
  default:
   throw cms::Exception("PFECALSuperClusterProducer::calculateRegressedEnergy")
     << "Supercluster seed is either EB nor EE!" << std::endl;
  }
  return -1;
}


