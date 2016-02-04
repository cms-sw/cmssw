#include "HLTrigger/special/interface/HLTPi0RecHitsFilter.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"


//Level 1 Trigger
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

/// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"


// Ecal Mapping 
#include "DataFormats/EcalRawData/interface/EcalListOfFEDS.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

// Jets stuff
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

//// Ecal Electrons Id
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"


// ES stuff
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"

//Ecal status
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"



#include "TVector3.h"

#define TWOPI 6.283185308

using namespace l1extra;
using namespace edm;
using namespace std;
using namespace trigger;


HLTPi0RecHitsFilter::HLTPi0RecHitsFilter(const edm::ParameterSet& iConfig)
{
  barrelHits_ = iConfig.getParameter< edm::InputTag > ("barrelHits");
  endcapHits_ = iConfig.getParameter< edm::InputTag > ("endcapHits");
  
  clusSeedThr_ = iConfig.getParameter<double> ("clusSeedThr");
  clusSeedThrEndCap_ = iConfig.getParameter<double> ("clusSeedThrEndCap");

  clusEtaSize_ = iConfig.getParameter<int> ("clusEtaSize");
  clusPhiSize_ = iConfig.getParameter<int> ("clusPhiSize");
  if ( clusPhiSize_ % 2 == 0 ||  clusEtaSize_ % 2 == 0) {
    edm::LogError("AlCaPi0RecHitsProducerError") << "Size of eta/phi for simple clustering should be odd numbers, reset to be 3 ";
    clusPhiSize_ = 3; 
    clusEtaSize_ = 3; 
  }
  
  //  seleNRHMax_ = iConfig.getParameter<int> ("seleNRHMax");
  // seleXtalMinEnergy_ = iConfig.getParameter<double>("seleXtalMinEnergy");
  // seleXtalMinEnergyEndCap_ = iConfig.getParameter<double>("seleXtalMinEnergyEndCap");
  useRecoFlag_ = iConfig.getParameter<bool>("useRecoFlag");
  flagLevelRecHitsToUse_ = iConfig.getParameter<int>("flagLevelRecHitsToUse"); 
  
  useDBStatus_ = iConfig.getParameter<bool>("useDBStatus");
  statusLevelRecHitsToUse_ = iConfig.getParameter<int>("statusLevelRecHitsToUse"); 
  
  nMinRecHitsSel1stCluster_ = iConfig.getParameter<int>("nMinRecHitsSel1stCluster");
  nMinRecHitsSel2ndCluster_ = iConfig.getParameter<int>("nMinRecHitsSel2ndCluster");
  
  maxNumberofSeeds_    = iConfig.getParameter<unsigned int> ("maxNumberofSeeds");
  maxNumberofClusters_ = iConfig.getParameter<unsigned int> ("maxNumberofClusters");
  
  doSelForPi0Barrel_ = iConfig.getParameter<bool> ("doSelForPi0Barrel");  
  if(doSelForPi0Barrel_){
    ///for Pi0 barrel selection
    selePtGamma_ = iConfig.getParameter<double> ("selePtGamma");  
    selePtPi0_ = iConfig.getParameter<double> ("selePtPi0");  
    seleMinvMaxPi0_ = iConfig.getParameter<double> ("seleMinvMaxPi0");  
    seleMinvMinPi0_ = iConfig.getParameter<double> ("seleMinvMinPi0");  
    seleS4S9Gamma_ = iConfig.getParameter<double> ("seleS4S9Gamma");  
    selePi0Iso_ = iConfig.getParameter<double> ("selePi0Iso");  
    ptMinForIsolation_ = iConfig.getParameter<double> ("ptMinForIsolation");
    selePi0BeltDR_ = iConfig.getParameter<double> ("selePi0BeltDR");  
    selePi0BeltDeta_ = iConfig.getParameter<double> ("selePi0BeltDeta");  

    storeIsoClusRecHitPi0EB_ = iConfig.getParameter<bool> ("storeIsoClusRecHitPi0EB");
    pi0BarrelHits_ = iConfig.getParameter< std::string > ("pi0BarrelHitCollection");
    
  }
  
  doSelForPi0Endcap_ = iConfig.getParameter<bool>("doSelForPi0Endcap");  
  if(doSelForPi0Endcap_){
    ///for Pi0 endcap selection
    ///    selePtGammaEndCap_ = iConfig.getParameter<double> ("selePtGammaEndCap");  
    /// selePtPi0EndCap_ = iConfig.getParameter<double> ("selePtPi0EndCap");   


    ///try to divide endcap region into 3 parts
    /// eta< 2 ; eta>2 && eta<2.5 ; eta>2.5; 
    region1_Pi0EndCap_ = iConfig.getParameter<double> ("region1_Pi0EndCap");
    selePtGammaPi0EndCap_region1_ = iConfig.getParameter<double> ("selePtGammaPi0EndCap_region1");  
    selePtPi0EndCap_region1_ = iConfig.getParameter<double> ("selePtPi0EndCap_region1");   
    
    //    preScale_endcapPi0_region1_ = iConfig.getParameter<int> ("preScale_endcapPi0_region1"); 
    // preScale_endcapPi0_region2_ = iConfig.getParameter<int> ("preScale_endcapPi0_region2"); 
    //preScale_endcapPi0_region3_ = iConfig.getParameter<int> ("preScale_endcapPi0_region3"); 
    
    
    
    region2_Pi0EndCap_ = iConfig.getParameter<double> ("region2_Pi0EndCap");
    selePtGammaPi0EndCap_region2_ = iConfig.getParameter<double> ("selePtGammaPi0EndCap_region2");  
    selePtPi0EndCap_region2_ = iConfig.getParameter<double> ("selePtPi0EndCap_region2");   
    
    selePtGammaPi0EndCap_region3_ = iConfig.getParameter<double> ("selePtGammaPi0EndCap_region3");  
    selePtPi0EndCap_region3_ = iConfig.getParameter<double> ("selePtPi0EndCap_region3"); 
    
    
    

    seleS4S9GammaEndCap_ = iConfig.getParameter<double> ("seleS4S9GammaEndCap");  
    seleMinvMaxPi0EndCap_ = iConfig.getParameter<double> ("seleMinvMaxPi0EndCap");  
    seleMinvMinPi0EndCap_ = iConfig.getParameter<double> ("seleMinvMinPi0EndCap");  
    ptMinForIsolationEndCap_ = iConfig.getParameter<double> ("ptMinForIsolationEndCap");
    selePi0BeltDREndCap_ = iConfig.getParameter<double> ("selePi0BeltDREndCap");  
    selePi0BeltDetaEndCap_ = iConfig.getParameter<double> ("selePi0BeltDetaEndCap");  
    selePi0IsoEndCap_ = iConfig.getParameter<double> ("selePi0IsoEndCap");  
    storeIsoClusRecHitPi0EE_ = iConfig.getParameter<bool> ("storeIsoClusRecHitPi0EE");
    pi0EndcapHits_ = iConfig.getParameter< std::string > ("pi0EndcapHitCollection");
  }
  
    
  
  doSelForEtaBarrel_ = iConfig.getParameter<bool>("doSelForEtaBarrel");  
  if(doSelForEtaBarrel_){
    ///for Eta barrel selection
    selePtGammaEta_ = iConfig.getParameter<double> ("selePtGammaEta");  
    selePtEta_ = iConfig.getParameter<double> ("selePtEta");   
    seleS4S9GammaEta_ = iConfig.getParameter<double> ("seleS4S9GammaEta");  
    seleS9S25GammaEta_ = iConfig.getParameter<double> ("seleS9S25GammaEta");  
    seleMinvMaxEta_ = iConfig.getParameter<double> ("seleMinvMaxEta");  
    seleMinvMinEta_ = iConfig.getParameter<double> ("seleMinvMinEta");  
    ptMinForIsolationEta_ = iConfig.getParameter<double> ("ptMinForIsolationEta");
    seleEtaIso_ = iConfig.getParameter<double> ("seleEtaIso");  
    seleEtaBeltDR_ = iConfig.getParameter<double> ("seleEtaBeltDR");  
    seleEtaBeltDeta_ = iConfig.getParameter<double> ("seleEtaBeltDeta");  
    storeIsoClusRecHitEtaEB_ = iConfig.getParameter<bool> ("storeIsoClusRecHitEtaEB");
    removePi0CandidatesForEta_ = iConfig.getParameter<bool>("removePi0CandidatesForEta");
    if(removePi0CandidatesForEta_){
      massLowPi0Cand_ = iConfig.getParameter<double>("massLowPi0Cand");
      massHighPi0Cand_ = iConfig.getParameter<double>("massHighPi0Cand");
    }
    etaBarrelHits_ = iConfig.getParameter< std::string > ("etaBarrelHitCollection");
    store5x5RecHitEtaEB_ = iConfig.getParameter<bool> ("store5x5RecHitEtaEB");
    store5x5IsoClusRecHitEtaEB_ = iConfig.getParameter<bool> ("store5x5IsoClusRecHitEtaEB");    
    
  }
  
  
  doSelForEtaEndcap_ = iConfig.getParameter<bool>("doSelForEtaEndcap");  
  if(doSelForEtaEndcap_){
    ///for Eta endcap selection
    ///  selePtGammaEtaEndCap_ = iConfig.getParameter<double> ("selePtGammaEtaEndCap");  
    ///selePtEtaEndCap_ = iConfig.getParameter<double> ("selePtEtaEndCap");   
    
    ///try to divide endcap region into 3 parts
    /// eta< 2 ; eta>2 && eta<2.5 ; eta>2.5; 
    region1_EtaEndCap_ = iConfig.getParameter<double> ("region1_EtaEndCap");
    selePtGammaEtaEndCap_region1_ = iConfig.getParameter<double> ("selePtGammaEtaEndCap_region1");  
    selePtEtaEndCap_region1_ = iConfig.getParameter<double> ("selePtEtaEndCap_region1");   
    
    region2_EtaEndCap_ = iConfig.getParameter<double> ("region2_EtaEndCap");
    selePtGammaEtaEndCap_region2_ = iConfig.getParameter<double> ("selePtGammaEtaEndCap_region2");  
    selePtEtaEndCap_region2_ = iConfig.getParameter<double> ("selePtEtaEndCap_region2");   
    
    selePtGammaEtaEndCap_region3_ = iConfig.getParameter<double> ("selePtGammaEtaEndCap_region3");  
    selePtEtaEndCap_region3_ = iConfig.getParameter<double> ("selePtEtaEndCap_region3");  
    
    //    preScale_endcapEta_region1_ = iConfig.getParameter<int> ("preScale_endcapEta_region1"); 
    // preScale_endcapEta_region2_ = iConfig.getParameter<int> ("preScale_endcapEta_region2"); 
    //preScale_endcapEta_region3_ = iConfig.getParameter<int> ("preScale_endcapEta_region3"); 
    
    
    seleS4S9GammaEtaEndCap_ = iConfig.getParameter<double> ("seleS4S9GammaEtaEndCap");  
    seleS9S25GammaEtaEndCap_ = iConfig.getParameter<double> ("seleS9S25GammaEtaEndCap");  
    seleMinvMaxEtaEndCap_ = iConfig.getParameter<double> ("seleMinvMaxEtaEndCap");  
    seleMinvMinEtaEndCap_ = iConfig.getParameter<double> ("seleMinvMinEtaEndCap");  
    ptMinForIsolationEtaEndCap_ = iConfig.getParameter<double> ("ptMinForIsolationEtaEndCap");
    seleEtaIsoEndCap_ = iConfig.getParameter<double> ("seleEtaIsoEndCap");  
    seleEtaBeltDREndCap_ = iConfig.getParameter<double> ("seleEtaBeltDREndCap");  
    seleEtaBeltDetaEndCap_ = iConfig.getParameter<double> ("seleEtaBeltDetaEndCap");  
    storeIsoClusRecHitEtaEE_ = iConfig.getParameter<bool> ("storeIsoClusRecHitEtaEE");
    etaEndcapHits_ = iConfig.getParameter< std::string > ("etaEndcapHitCollection");
    store5x5RecHitEtaEE_ = iConfig.getParameter<bool> ("store5x5RecHitEtaEE");
    store5x5IsoClusRecHitEtaEE_ = iConfig.getParameter<bool> ("store5x5IsoClusRecHitEtaEE");
  }
  
  
  preshHitProducer_   = iConfig.getParameter<edm::InputTag>("preshRecHitProducer");
  ///for storing rechits ES for each selected EE clusters.
  storeRecHitES_ = iConfig.getParameter<bool>("storeRecHitES");  
  if(storeRecHitES_){
    // maximum number of matched ES clusters (in each ES layer) to each BC
    preshNclust_             = iConfig.getParameter<int>("preshNclust");
    // min energy of ES clusters
    preshClustECut = iConfig.getParameter<double>("preshClusterEnergyCut");
    // algo params
    float preshStripECut = iConfig.getParameter<double>("preshStripEnergyCut");
    int preshSeededNst = iConfig.getParameter<int>("preshSeededNstrip");
    // calibration parameters:
    calib_planeX_ = iConfig.getParameter<double>("preshCalibPlaneX");
    calib_planeY_ = iConfig.getParameter<double>("preshCalibPlaneY");
    gamma_        = iConfig.getParameter<double>("preshCalibGamma");
    mip_          = iConfig.getParameter<double>("preshCalibMIP");

    // The debug level
    std::string debugString = iConfig.getParameter<std::string>("debugLevelES");
    // ES algo constructor:
    presh_algo = new PreshowerClusterAlgo(preshStripECut,preshClustECut,preshSeededNst);

    if(doSelForPi0Endcap_){
      pi0ESHits_ = iConfig.getParameter< std::string > ("pi0ESCollection");
    }
    if(doSelForEtaEndcap_){
      etaESHits_ = iConfig.getParameter< std::string > ("etaESCollection");
    }
    
  }
  
  ///  l1IsolatedTag_ = iConfig.getParameter< edm::InputTag > ("l1IsolatedTag");
  ///  l1NonIsolatedTag_ = iConfig.getParameter< edm::InputTag > ("l1NonIsolatedTag");
  //SB comment out, no usage in the following code
  //  l1SeedFilterTag_ = iConfig.getParameter< edm::InputTag > ("l1SeedFilterTag");

  debug_ = iConfig.getParameter<int> ("debugLevel");
  
   //Setup for core tools objects. 
  edm::ParameterSet posCalcParameters = iConfig.getParameter<edm::ParameterSet>("posCalcParameters");
  posCalculator_ = PositionCalc(posCalcParameters);

  //register your products
  doBarrel = true; 
  if(doSelForPi0Barrel_ && doSelForEtaBarrel_){
    BarrelHits_ = pi0BarrelHits_;
  }
  else if(!doSelForPi0Barrel_ && doSelForEtaBarrel_){
    BarrelHits_ = etaBarrelHits_;
  }
  else if(doSelForPi0Barrel_ && !doSelForEtaBarrel_){
    BarrelHits_ = pi0BarrelHits_;
  }else{
    doBarrel = false; 
  }
  
  if(doBarrel){
    produces< EBRecHitCollection >(BarrelHits_);
  }
  
  
  
  doEndcap = true; 
  if(doSelForPi0Endcap_ && doSelForEtaEndcap_){
    EndcapHits_ = pi0EndcapHits_;
    ESHits_ = pi0ESHits_;
  }else if(!doSelForPi0Endcap_ && doSelForEtaEndcap_){
    EndcapHits_ = etaEndcapHits_;
    ESHits_ = etaESHits_;
  }else if(doSelForPi0Endcap_ && !doSelForEtaEndcap_){
    EndcapHits_ = pi0EndcapHits_;
    ESHits_ = pi0ESHits_;
  }else{
    doEndcap = false; 
  }
  
  if(doEndcap){
    produces< EERecHitCollection >(EndcapHits_);
    if( storeRecHitES_)  produces< ESRecHitCollection >(ESHits_);
  }
    
  
}





HLTPi0RecHitsFilter::~HLTPi0RecHitsFilter()
{
  //delete TheMapping;
  
  if(storeRecHitES_){
    delete presh_algo;
  }
}


// ------------ method called to produce the data  ------------
bool
HLTPi0RecHitsFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    
  
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle); 
  const CaloSubdetectorGeometry *geometry_eb = geoHandle->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
  const CaloSubdetectorGeometry *geometry_ee = geoHandle->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
  const CaloSubdetectorGeometry *geometry_es = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  
  edm::ESHandle<CaloTopology> pTopology;
  iSetup.get<CaloTopologyRecord>().get(pTopology);
  const CaloSubdetectorTopology *topology_eb = pTopology->getSubdetectorTopology(DetId::Ecal,EcalBarrel);
  const CaloSubdetectorTopology *topology_ee = pTopology->getSubdetectorTopology(DetId::Ecal,EcalEndcap);
  
  CaloSubdetectorTopology *topology_es=0;
  if (geometry_es) {
    topology_es  = new EcalPreshowerTopology(geoHandle);
  }else{
    storeRecHitES_ = false; ///if no preshower
  }
  
  
  
  // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  iSetup.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;

  vector<int>::iterator it; 
  
  
  
  ///get status from DB
  edm::ESHandle<EcalChannelStatus> csHandle;
  if ( useDBStatus_ ) iSetup.get<EcalChannelStatusRcd>().get(csHandle);
  const EcalChannelStatus &channelStatus = *csHandle; 
  
  
  ///==============Start to process barrel part==================///
    
  Handle<EBRecHitCollection> barrelRecHitsHandle;
  iEvent.getByLabel(barrelHits_,barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    LogDebug("") << "AlCaPi0RecHitsProducer: Error! can't get product barrel hits!" << std::endl;
  }
  
  const EcalRecHitCollection *hitCollection_p = barrelRecHitsHandle.product();


  if(debug_>=1) std::cout<<" barrel_input_size: "<<iEvent.id().run()<<" "<<iEvent.id().event()<<" "<<hitCollection_p->size()<<std::endl;
  
  
  
  std::vector<EcalRecHit> seeds;
  seeds.clear();

  vector<EBDetId> usedXtals;
  usedXtals.clear();
  
  detIdEBRecHits.clear(); //// EBDetId
  EBRecHits.clear();  /// EcalRecHit
  
  ////make seeds. 
  EBRecHitCollection::const_iterator itb;
  for (itb=barrelRecHitsHandle->begin(); itb!=barrelRecHitsHandle->end(); itb++) {
    double energy = itb->energy();

    //if( energy < seleXtalMinEnergy_) continue; 
    
    
    if(useRecoFlag_ ){ ///from recoFlag()
      int flag = itb->recoFlag();
      if( flagLevelRecHitsToUse_ ==0){ ///good 
	if( flag != 0) continue; 
      }
      else if( flagLevelRecHitsToUse_ ==1){ ///good || PoorCalib 
	if( flag !=0 && flag != 4 ) continue; 
      }
      else if( flagLevelRecHitsToUse_ ==2){ ///good || PoorCalib || LeadingEdgeRecovered || kNeighboursRecovered,
	if( flag !=0 && flag != 4 && flag != 6 && flag != 7) continue; 
      }
    }
    if ( useDBStatus_){ //// from DB
      int status =  int(channelStatus[itb->id().rawId()].getStatusCode()); 
      if ( status > statusLevelRecHitsToUse_ ) continue; 
    }
    
    
    EBDetId det = itb->id();
    
    detIdEBRecHits.push_back(det);
    EBRecHits.push_back(*itb);
    
    
    if (energy > clusSeedThr_) {
      seeds.push_back(*itb);
      if( seeds.size() > maxNumberofSeeds_) return false; 
    }
  }
  
  
  
  
  
  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > selEBRecHitCollection( new EBRecHitCollection );
  std::auto_ptr< EERecHitCollection > selEERecHitCollection( new EERecHitCollection );
  vector<DetId> selectedEBDetIds;
  vector<DetId> selectedEEDetIds; 
    

  int nClus;
  vector<float> eClus;
  vector<float> etClus;
  vector<float> etaClus;
  vector<float> thetaClus;
  vector<float> phiClus;
  vector<EBDetId> max_hit;
  vector< vector<EcalRecHit> > RecHitsCluster;
  vector< vector<EcalRecHit> > RecHitsCluster5x5;
  vector<float> s4s9Clus;
  vector<float> s9s25Clus;
  
  nClus=0;

  // Make own simple clusters (3x3, 5x5 or clusPhiSize_ x clusEtaSize_)
  sort(seeds.begin(), seeds.end(), ecalRecHitLess());

  for (std::vector<EcalRecHit>::iterator itseed=seeds.begin(); itseed!=seeds.end(); itseed++) {
    EBDetId seed_id = itseed->id();
    std::vector<EBDetId>::const_iterator usedIds;
    
    bool seedAlreadyUsed=false;
    for(usedIds=usedXtals.begin(); usedIds!=usedXtals.end(); usedIds++){
      if(*usedIds==seed_id){
	seedAlreadyUsed=true;
	break; 
      }
    }

    if(seedAlreadyUsed)continue;

    std::vector<DetId> clus_v = topology_eb->getWindow(seed_id,clusEtaSize_,clusPhiSize_);	
    std::vector<std::pair<DetId, float> > clus_used;
    

    vector<EcalRecHit> RecHitsInWindow;
    vector<EcalRecHit> RecHitsInWindow5x5;
    float simple_energy = 0; 
    
    for (std::vector<DetId>::iterator det=clus_v.begin(); det!=clus_v.end(); det++) {
      EBDetId EBdet = *det;
      
      bool  HitAlreadyUsed=false;
      for(usedIds=usedXtals.begin(); usedIds!=usedXtals.end(); usedIds++){
	if(*usedIds==*det){
	  HitAlreadyUsed=true;
	  break;
	}
      }
     
      if(HitAlreadyUsed)continue;
      
      std::vector<EBDetId>::iterator itdet = find( detIdEBRecHits.begin(),detIdEBRecHits.end(),EBdet);
      if(itdet == detIdEBRecHits.end()) continue; 
      
      int nn = int(itdet - detIdEBRecHits.begin());
      usedXtals.push_back(*det);
      RecHitsInWindow.push_back(EBRecHits[nn]);
      clus_used.push_back(std::pair<DetId, float>(*det, 1) );
      simple_energy = simple_energy + EBRecHits[nn].energy();
      
            
    }
    
    if(simple_energy <= 0) continue; 
    
    
    math::XYZPoint clus_pos = posCalculator_.Calculate_Location(clus_used,hitCollection_p,geometry_eb,geometry_es);
    
    
    float theta_s = 2. * atan(exp(-clus_pos.eta()));
    float et_s = simple_energy * sin(theta_s);
    
  
    //Compute S4/S9 variable
    //We are not sure to have 9 RecHits so need to check eta and phi:

    ///check s4s9
    float s4s9_tmp[4];
    for(int i=0;i<4;i++)s4s9_tmp[i]= 0;

    int seed_ieta = seed_id.ieta();
    int seed_iphi = seed_id.iphi();
    
    convxtalid( seed_iphi,seed_ieta);
    
    float e3x3 = 0; 
    float e5x5 = 0; 
    for(unsigned int j=0; j<RecHitsInWindow.size();j++){
      EBDetId det = (EBDetId)RecHitsInWindow[j].id(); 
      
      int ieta = det.ieta();
      int iphi = det.iphi();
      
      convxtalid(iphi,ieta);
      
      float en = RecHitsInWindow[j].energy(); 
      
      int dx = diff_neta_s(seed_ieta,ieta);
      int dy = diff_nphi_s(seed_iphi,iphi);
    
      
      if(std::abs(dx)<=1 && std::abs(dy)<=1) {
	e3x3 += en; 
	if(dx <= 0 && dy <=0) s4s9_tmp[0] += en; 
	if(dx >= 0 && dy <=0) s4s9_tmp[1] += en; 
	if(dx <= 0 && dy >=0) s4s9_tmp[2] += en; 
	if(dx >= 0 && dy >=0) s4s9_tmp[3] += en; 
      }
    }
    

    if(e3x3 <= 0) continue; 
    
    

    float s4s9_max = *max_element( s4s9_tmp,s4s9_tmp+4)/e3x3; 
   
    
    ///calculate e5x5
    std::vector<DetId> clus_v5x5 = topology_eb->getWindow(seed_id,5,5);	
    for( std::vector<DetId>::const_iterator idItr = clus_v5x5.begin(); idItr != clus_v5x5.end(); idItr++){
      EBDetId det = *idItr;
      //inside collections
      std::vector<EBDetId>::iterator itdet = find( detIdEBRecHits.begin(),detIdEBRecHits.end(),det);
      if(itdet == detIdEBRecHits.end()) continue; 
      int nn = int(itdet - detIdEBRecHits.begin());
      
      RecHitsInWindow5x5.push_back(EBRecHits[nn]);
      e5x5 += EBRecHits[nn].energy();
      
    }
    
    
    if(e5x5 <= 0) continue; 
    
    eClus.push_back(simple_energy);
    etClus.push_back(et_s);
    etaClus.push_back(clus_pos.eta());
    thetaClus.push_back(theta_s);
    phiClus.push_back(clus_pos.phi());
    s4s9Clus.push_back(s4s9_max);
    s9s25Clus.push_back(e3x3/e5x5);
    RecHitsCluster.push_back(RecHitsInWindow);
    RecHitsCluster5x5.push_back(RecHitsInWindow5x5);
    
    if(debug_>=1){
      cout<<"3x3_cluster_eb (n,nxt,e,et eta,phi,s4s9,s9s25) "<<nClus<<" "<<int(RecHitsInWindow.size())<<" "<<eClus[nClus]<<" "<<etClus[nClus]<<" "<<etaClus[nClus]<<" "<<phiClus[nClus]<<" "<<s4s9Clus[nClus]<<" "
	  <<s9s25Clus[nClus]<<endl;
    }
    
    nClus++;
    if( nClus > (int) maxNumberofClusters_) return false; 
    
  }
  

  // Selection, based on Simple clustering
  //pi0 candidates
  
  int npi0_s=0;
  ////to avoid duplicated push_back rechit
  vector<int> indClusSelected; 

  ///do selection for pi0->gg barrel
  if(doSelForPi0Barrel_){
    

    for(int i=0 ; i<nClus ; i++){
      for(int j=i+1 ; j<nClus ; j++){
      
	if( etClus[i]>selePtGamma_ && etClus[j]>selePtGamma_ && s4s9Clus[i]>seleS4S9Gamma_ && s4s9Clus[j]>seleS4S9Gamma_){
        
	  float p0x = etClus[i] * cos(phiClus[i]);
	  float p1x = etClus[j] * cos(phiClus[j]);
	  float p0y = etClus[i] * sin(phiClus[i]);
	  float p1y = etClus[j] * sin(phiClus[j]);
	  float p0z = eClus[i] * cos(thetaClus[i]);
	  float p1z = eClus[j] * cos(thetaClus[j]);
	
        
	  float pt_pair = sqrt( (p0x+p1x)*(p0x+p1x) + (p0y+p1y)*(p0y+p1y));

	  if (pt_pair < selePtPi0_)continue;
	  float m_inv = sqrt ( (eClus[i] + eClus[j])*(eClus[i] + eClus[j]) - (p0x+p1x)*(p0x+p1x) - (p0y+p1y)*(p0y+p1y) - (p0z+p1z)*(p0z+p1z) );  
	  
	  if ( (m_inv<seleMinvMaxPi0_) && (m_inv>seleMinvMinPi0_) ){
	  
	  
	    //New Loop on cluster to measure isolation:
	    vector<int> IsoClus;
	    IsoClus.clear();
	    float Iso = 0;
	    TVector3 pairVect = TVector3((p0x+p1x), (p0y+p1y), (p0z+p1z));
	    for(int k=0 ; k<nClus ; k++){

	      if(etClus[k] < ptMinForIsolation_) continue; 
	    
	    
	      if(k==i || k==j)continue;
	      
	      TVector3 clusVect = TVector3(etClus[k] *cos(phiClus[k]), etClus[k] * sin(phiClus[k]) , eClus[k] * cos(thetaClus[k]));
	      float dretacl = fabs(etaClus[k] - pairVect.Eta());
	      float drcl = clusVect.DeltaR(pairVect);
	      
	      if((drcl<selePi0BeltDR_) && (dretacl<selePi0BeltDeta_) ){
		Iso = Iso + etClus[k];
		IsoClus.push_back(k);
	      }
	    }
	    
	    if(Iso/pt_pair<selePi0Iso_){
	      ///additional check on # of rechits in the selcted clusters
	      if( int(RecHitsCluster[i].size()) >= nMinRecHitsSel1stCluster_ && int(RecHitsCluster[j].size()) >= nMinRecHitsSel2ndCluster_ ){
		
		int indtmp[2]={i,j};
		for(int jj =0; jj<2; jj++){
		  int ind = indtmp[jj];
		  it = find(indClusSelected.begin(),indClusSelected.end(),ind);
		  if( it == indClusSelected.end()){
		    indClusSelected.push_back(ind);
		    for(unsigned int Rec=0;Rec<RecHitsCluster[ind].size();Rec++) {
		      selEBRecHitCollection->push_back(RecHitsCluster[ind][Rec]);
		      selectedEBDetIds.push_back(RecHitsCluster[ind][Rec].id());
		    }
		  }
		}
		
		if( storeIsoClusRecHitPi0EB_){
		  
		  for(unsigned int iii=0 ; iii<IsoClus.size() ; iii++){   
		    int ind = IsoClus[iii];
		    it = find(indClusSelected.begin(),indClusSelected.end(),ind);
		    if( it == indClusSelected.end()){
		      indClusSelected.push_back(ind);
		      for(unsigned int Rec3=0;Rec3<RecHitsCluster[ind].size();Rec3++) {
			selEBRecHitCollection->push_back(RecHitsCluster[ind][Rec3]);
			selectedEBDetIds.push_back(RecHitsCluster[ind][Rec3].id());
		    }
		    }
		  } 
		}
		
		npi0_s++;
	      } ///number of rechits passed
	    } //isolation passed
	    
	    ///	    if(npi0_s == MAXPI0S) return false; 
	  } ///mass window passed
	} /// PT and S4/S9 passed
      } // End of the "j" loop over Simple Clusters
    } // End of the "i" loop over Simple Clusters
    
    
    if(debug_>=1) cout<<"npi0seleb: "<<npi0_s<<endl;
    
    
    
    
  }///end of selection on pi0->gg in in barrel
    

  ///do selection for eta->gg in barrel
  if(doSelForEtaBarrel_){

    vector<int> indEtaCand; 

    vector<int> indClusPi0Candidates; 
    if( removePi0CandidatesForEta_){
      
      for(int i=0 ; i<nClus ; i++){
	for(int j=i+1 ; j<nClus ; j++){
	  
	  float p0x = etClus[i] * cos(phiClus[i]);
	  float p1x = etClus[j] * cos(phiClus[j]);
	  float p0y = etClus[i] * sin(phiClus[i]);
	  float p1y = etClus[j] * sin(phiClus[j]);
	  float p0z = eClus[i] * cos(thetaClus[i]);
	  float p1z = eClus[j] * cos(thetaClus[j]);
	  float m_inv = sqrt ( (eClus[i] + eClus[j])*(eClus[i] + eClus[j]) - (p0x+p1x)*(p0x+p1x) - (p0y+p1y)*(p0y+p1y) - (p0z+p1z)*(p0z+p1z) );  
	  
	  int tmp[2] = {i,j};
	  
	  if(m_inv > massLowPi0Cand_ && m_inv < massHighPi0Cand_){
	    for( int k=0;k<2; k++){
	      it = find(indClusPi0Candidates.begin(),indClusPi0Candidates.end(),tmp[k]);
	      if( it == indClusPi0Candidates.end()) indClusPi0Candidates.push_back(tmp[k]);
	      
	    }
	  }
	  
	}
      }
      
    }
    
    

    for(int i=0 ; i<nClus ; i++){
      for(int j=i+1 ; j<nClus ; j++){
	
	if( removePi0CandidatesForEta_){
	  int tmp[2] = {i,j};
	  int flagPi0 = 0; 
	  for( int k=0;k<2; k++){
	    it = find(indClusPi0Candidates.begin(),indClusPi0Candidates.end(),tmp[k]);
	    if( it != indClusPi0Candidates.end())  {
	      flagPi0 = 1; 
	      break; 
	    }
	  }
	  if(flagPi0==1) continue; 
	}
		

	if( etClus[i]>selePtGammaEta_ && etClus[j]>selePtGammaEta_ && s4s9Clus[i]>seleS4S9GammaEta_ && s4s9Clus[j]>seleS4S9GammaEta_
	    && s9s25Clus[i]>seleS9S25GammaEta_ && s9s25Clus[j]>seleS9S25GammaEta_
	    ){
	  float p0x = etClus[i] * cos(phiClus[i]);
	  float p1x = etClus[j] * cos(phiClus[j]);
	  float p0y = etClus[i] * sin(phiClus[i]);
	  float p1y = etClus[j] * sin(phiClus[j]);
	  float p0z = eClus[i] * cos(thetaClus[i]);
	  float p1z = eClus[j] * cos(thetaClus[j]);
	  
	  float pt_pair = sqrt( (p0x+p1x)*(p0x+p1x) + (p0y+p1y)*(p0y+p1y));
	  if (pt_pair < selePtEta_ ) continue;
	  
	  float m_inv = sqrt ( (eClus[i] + eClus[j])*(eClus[i] + eClus[j]) - (p0x+p1x)*(p0x+p1x) - (p0y+p1y)*(p0y+p1y) - (p0z+p1z)*(p0z+p1z) );  
	  
	  if ( (m_inv<seleMinvMaxEta_) && (m_inv>seleMinvMinEta_) ){

	    //New Loop on cluster to measure isolation:
	    vector<int> IsoClus;
	    IsoClus.clear();
	    float Iso = 0;
	    TVector3 pairVect = TVector3((p0x+p1x), (p0y+p1y), (p0z+p1z));
	    for(int k=0 ; k<nClus ; k++){
	      
	      if(etClus[k] < ptMinForIsolationEta_) continue; 
	    
	      if(k==i || k==j)continue;
	      TVector3 clusVect = TVector3(etClus[k] * cos(phiClus[k]), etClus[k] * sin(phiClus[k]) , eClus[k] * cos(thetaClus[k]));
	      float dretacl = fabs(etaClus[k] - pairVect.Eta());
	      float drcl = clusVect.DeltaR(pairVect);
	      if((drcl<seleEtaBeltDR_) && (dretacl<seleEtaBeltDeta_) ){
		Iso = Iso + etClus[k];
		IsoClus.push_back(k);
	      }
	    }
	    
	    if(Iso/pt_pair < seleEtaIso_){
	      
	      if( int(RecHitsCluster[i].size()) >= nMinRecHitsSel1stCluster_ && int(RecHitsCluster[j].size()) >= nMinRecHitsSel2ndCluster_ ){
		
		int indtmp[2]={i,j};
		for(int jj =0; jj<2; jj++){
		  int ind = indtmp[jj];
		
		  ///eta candidates
		  it = find(indEtaCand.begin(),indEtaCand.end(),ind);
		  if(it == indEtaCand.end()){
		    indEtaCand.push_back(ind);
		  }
		
		  it = find(indClusSelected.begin(),indClusSelected.end(),ind);
		  if( it == indClusSelected.end()){
		    indClusSelected.push_back(ind);
		    for(unsigned int Rec=0;Rec<RecHitsCluster[ind].size();Rec++) {
		      selEBRecHitCollection->push_back(RecHitsCluster[ind][Rec]);
		      selectedEBDetIds.push_back(RecHitsCluster[ind][Rec].id());
		    }
		  
		  }
		}
	      
	      
		if( storeIsoClusRecHitEtaEB_){
		  for(unsigned int iii=0 ; iii<IsoClus.size() ; iii++){   
		    int ind = IsoClus[iii];

		    if(store5x5IsoClusRecHitEtaEB_){
		      ///eta candidates isoClus.
		      it = find(indEtaCand.begin(),indEtaCand.end(),ind);
		      if(it == indEtaCand.end()){
			indEtaCand.push_back(ind);
		      }
		    }

		    it = find(indClusSelected.begin(),indClusSelected.end(),ind);
		    if( it == indClusSelected.end()){
		      indClusSelected.push_back(ind);
		      for(unsigned int Rec3=0;Rec3<RecHitsCluster[ind].size();Rec3++)  {
			selEBRecHitCollection->push_back(RecHitsCluster[ind][Rec3]);
			selectedEBDetIds.push_back(RecHitsCluster[ind][Rec3].id());
		      }
		    
		    }
		  } 
		}
	      } /// # of Rechits passed
	    } /// Isolation passed
	    
	  } /// Inside Eta Mass window
	  
	} //// PT Cut && S4S9 Cut satisfied.
	
	
      } // End of the "j" loop over Simple Clusters
    } // End of the "i" loop over Simple Clusters
    
    if( store5x5RecHitEtaEB_){
      ///for selected eta->gg candidates save 5x5 rechits also
      for(int j=0; j<int(indEtaCand.size());j++){
	int ind = indEtaCand[j];
	for(unsigned int Rec3=0;Rec3<RecHitsCluster5x5[ind].size();Rec3++) {
	
	  DetId det = RecHitsCluster5x5[ind][Rec3].id();
	  std::vector<DetId>::iterator itdet = find(selectedEBDetIds.begin(),selectedEBDetIds.end(),det);
	  if(itdet == selectedEBDetIds.end()){
	    selectedEBDetIds.push_back(det);
	    selEBRecHitCollection->push_back(RecHitsCluster5x5[ind][Rec3]);
	    
	  }
	
	}
      }
    }
    
  }////end of selections of eta->gg barrel
    
  
  
  if(debug_>=1) std::cout<<" barrel_output_size: "<<iEvent.id().run()<<" "<<iEvent.id().event()<<" "<<selEBRecHitCollection->size()<<std::endl;
  ///==============End of  barrel ==================///
  
  
  

  ///==============Start of  Endcap ==================///
  ///get preshower rechits
  
  Handle<ESRecHitCollection> esRecHitsHandle;
  iEvent.getByLabel(preshHitProducer_,esRecHitsHandle);
  if( !esRecHitsHandle.isValid()){
    LogDebug("") << "AlCaPi0RecHitsProducer: Error! can't get product esRecHit!" << std::endl;
  }
  const EcalRecHitCollection* hitCollection_es = esRecHitsHandle.product();
  // make a map of rechits:
  //  std::map<DetId, EcalRecHit> esrechits_map;
  esrechits_map.clear();
  EcalRecHitCollection::const_iterator iter;
  for (iter = esRecHitsHandle->begin(); iter != esRecHitsHandle->end(); iter++) {
    //Make the map of DetID, EcalRecHit pairs
    esrechits_map.insert(std::make_pair(iter->id(), *iter));   
  }
  // The set of used DetID's for a given event:
  //  std::set<DetId> used_strips;
  used_strips.clear();
  std::auto_ptr<ESRecHitCollection> selESRecHitCollection(new ESRecHitCollection );
  
  
  Handle<EERecHitCollection> endcapRecHitsHandle;
  iEvent.getByLabel(endcapHits_,endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    LogDebug("") << "AlCaPi0RecHitsProducer: Error! can't get product eeRecHit!" << std::endl;
  }
  
  const EcalRecHitCollection *hitCollection_e = endcapRecHitsHandle.product();
  if(debug_>=1) std::cout<<" endcap_input_size: "<<iEvent.id().run()<<" "<<iEvent.id().event()<<" "<<hitCollection_e->size()<<" "<<hitCollection_es->size()<<std::endl;
  
  
  
  detIdEERecHits.clear(); //// EEDetId
  EERecHits.clear();  /// EcalRecHit


  std::vector<EcalRecHit> seedsEndCap;
  seedsEndCap.clear();

  vector<EEDetId> usedXtalsEndCap;
  usedXtalsEndCap.clear();
  
  
  ////make seeds. 
  EERecHitCollection::const_iterator ite;
  for (ite=endcapRecHitsHandle->begin(); ite!=endcapRecHitsHandle->end(); ite++) {
    double energy = ite->energy();

    //if( energy < seleXtalMinEnergyEndCap_) continue; 

    if(  useRecoFlag_ ){ ///from recoFlag()
      int flag = ite->recoFlag();
      
      if( flagLevelRecHitsToUse_ ==0){ ///good 
	if( flag != 0) continue; 
      }
      else if( flagLevelRecHitsToUse_ ==1){ ///good || PoorCalib 
	if( flag !=0 && flag != 4 ) continue; 
      }
      else if( flagLevelRecHitsToUse_ ==2){ ///good || PoorCalib || LeadingEdgeRecovered || kNeighboursRecovered,
	if( flag !=0 && flag != 4 && flag != 6 && flag != 7) continue; 
      }
    }
    if( useDBStatus_) { //// from DB
      int status =  int(channelStatus[ite->id().rawId()].getStatusCode()); 
      if ( status > statusLevelRecHitsToUse_ ) continue; 
    }
    
    
    EEDetId det = ite->id();
    detIdEERecHits.push_back(det);
    EERecHits.push_back(*ite);
    
    
    if (energy > clusSeedThrEndCap_) {
      seedsEndCap.push_back(*ite);
      if( seedsEndCap.size() > maxNumberofSeeds_) return false;
    }
    
  }
  
  
  
  
  int nClusEndCap;
  vector<float> eClusEndCap;
  vector<float> etClusEndCap;
  vector<float> etaClusEndCap;
  vector<float> thetaClusEndCap;

  vector<float> xClusEndCap;
  vector<float> yClusEndCap;
  vector<float> zClusEndCap;

  
  vector<float> phiClusEndCap;
  vector< vector<EcalRecHit> > RecHitsClusterEndCap;
  vector< vector<EcalRecHit> > RecHitsCluster5x5EndCap;
  vector<float> s4s9ClusEndCap;
  vector<float> s9s25ClusEndCap;
  
  ///detid for each ee cluster, both in X and Y plane
  //  vector< vector<DetId> > esdetIDClusterEndCap;
  ///below are not necessary. 
  //vector<DetId> pi0_esdetid_stored;
  //vector<DetId> eta_esdetid_stored;
  
  nClusEndCap=0;
    
  
  
  // Make own simple clusters (3x3, 5x5 or clusPhiSize_ x clusEtaSize_)
  sort(seedsEndCap.begin(), seedsEndCap.end(), ecalRecHitLess());
  
  
  for (std::vector<EcalRecHit>::iterator itseed=seedsEndCap.begin(); itseed!=seedsEndCap.end(); itseed++) {
    EEDetId seed_id = itseed->id();
    std::vector<EEDetId>::const_iterator usedIds;
    
    bool seedAlreadyUsed=false;
    for(usedIds=usedXtalsEndCap.begin(); usedIds!=usedXtalsEndCap.end(); usedIds++){
      if(*usedIds==seed_id){
	seedAlreadyUsed=true;
	break; 
      }
    }

    if(seedAlreadyUsed)continue;

    std::vector<DetId> clus_v = topology_ee->getWindow(seed_id,clusEtaSize_,clusPhiSize_);	
    std::vector<std::pair<DetId, float> > clus_used;
    
    

    vector<EcalRecHit> RecHitsInWindow;
    vector<EcalRecHit> RecHitsInWindow5x5;
    
    float simple_energy = 0; 
    
    for (std::vector<DetId>::iterator det=clus_v.begin(); det!=clus_v.end(); det++) {
      EEDetId EEdet = *det;
      
      bool  HitAlreadyUsed=false;
      for(usedIds=usedXtalsEndCap.begin(); usedIds!=usedXtalsEndCap.end(); usedIds++){
	if(*usedIds==*det){
	  HitAlreadyUsed=true;
	  break;
	}
      }
     
      if(HitAlreadyUsed)continue;
      
      
      std::vector<EEDetId>::iterator itdet = find( detIdEERecHits.begin(),detIdEERecHits.end(),EEdet);
      if(itdet == detIdEERecHits.end()) continue; 
      
      
      int nn = int(itdet - detIdEERecHits.begin());
      usedXtalsEndCap.push_back(*det);
      RecHitsInWindow.push_back(EERecHits[nn]);
      clus_used.push_back(std::pair<DetId, float>(*det, 1) );
      simple_energy = simple_energy + EERecHits[nn].energy();
        
    }
    
    if( simple_energy <= 0) continue; 
    
    math::XYZPoint clus_pos = posCalculator_.Calculate_Location(clus_used,hitCollection_e,geometry_ee,geometry_es);
    
    float theta_s = 2. * atan(exp(-clus_pos.eta()));
    float et_s = simple_energy * sin(theta_s);
    
    //Compute S4/S9 variable
    //We are not sure to have 9 RecHits so need to check eta and phi:
    float s4s9_tmp[4];
    for(int i=0;i<4;i++) s4s9_tmp[i]= 0; 
    
    int ixSeed = seed_id.ix();
    int iySeed = seed_id.iy();
    float e3x3 = 0; 
    float e5x5 = 0;
    for(unsigned int j=0; j<RecHitsInWindow.size();j++){
      EEDetId det_this = (EEDetId)RecHitsInWindow[j].id(); 
      int dx = ixSeed - det_this.ix();
      int dy = iySeed - det_this.iy();
      
      float en = RecHitsInWindow[j].energy(); 
      if( std::abs(dx)<=1 && std::abs(dy)<=1) {
	e3x3 += en; 
	if(dx <= 0 && dy <=0) s4s9_tmp[0] += en; 
	if(dx >= 0 && dy <=0) s4s9_tmp[1] += en; 
	if(dx <= 0 && dy >=0) s4s9_tmp[2] += en; 
	if(dx >= 0 && dy >=0) s4s9_tmp[3] += en; 
      }
    }
    
    if(e3x3 <= 0) continue; 
    
    
    std::vector<DetId> clus_v5x5 = topology_ee->getWindow(seed_id,5,5);	
    for( std::vector<DetId>::const_iterator idItr = clus_v5x5.begin(); idItr != clus_v5x5.end(); idItr++){
      EEDetId det = *idItr;
      //inside collections
      std::vector<EEDetId>::iterator itdet = find( detIdEERecHits.begin(),detIdEERecHits.end(),det);
      if(itdet == detIdEERecHits.end()) continue; 
      int nn = int(itdet - detIdEERecHits.begin());
      
      RecHitsInWindow5x5.push_back(EERecHits[nn]);
      e5x5 +=  EERecHits[nn].energy();
      
    }
    
    if(e5x5 <= 0) continue; 
    
    
  
    xClusEndCap.push_back(clus_pos.x());
    yClusEndCap.push_back(clus_pos.y());
    zClusEndCap.push_back(clus_pos.z());

    etaClusEndCap.push_back(clus_pos.eta());
    thetaClusEndCap.push_back(theta_s);
    phiClusEndCap.push_back(clus_pos.phi());
    s4s9ClusEndCap.push_back(*max_element( s4s9_tmp,s4s9_tmp+4)/e3x3);
    s9s25ClusEndCap.push_back(e3x3/e5x5);
    RecHitsClusterEndCap.push_back(RecHitsInWindow);
    RecHitsCluster5x5EndCap.push_back(RecHitsInWindow5x5);
        
    
    eClusEndCap.push_back(simple_energy);
    etClusEndCap.push_back(et_s);
        
    
    if(debug_>=1){
      cout<<"3x3_cluster_ee (n,nxt,et eta,phi,s4s9,s925) "<<nClusEndCap<<" "<<int(RecHitsInWindow.size())<<" "<<eClusEndCap[nClusEndCap]<<" "<<" "<<etClusEndCap[nClusEndCap]<<" "<<etaClusEndCap[nClusEndCap]<<" "<<phiClusEndCap[nClusEndCap]<<" "<<s4s9ClusEndCap[nClusEndCap]<<" "<<s9s25ClusEndCap[nClusEndCap]<<endl;
    }
    
    nClusEndCap++;

    if( nClusEndCap > (int) maxNumberofClusters_) return false; 
    
    ///    if (nClusEndCap == MAXCLUS) return false; 
  }
  
  
  
  // Selection, based on Simple clustering 
  int npi0_se=0;
  ////to avoid duplicated push_back rechit
  vector<int> indClusEndCapSelected; 
  
  
  if(doSelForPi0Endcap_){


    for(int i=0 ; i<nClusEndCap ; i++){
      for(int j=i+1 ; j<nClusEndCap ; j++){
      

	if( s4s9ClusEndCap[i] < seleS4S9GammaEndCap_ || s4s9ClusEndCap[j] < seleS4S9GammaEndCap_ ) continue; 
	float p0x = etClusEndCap[i] * cos(phiClusEndCap[i]);
	float p1x = etClusEndCap[j] * cos(phiClusEndCap[j]);
	float p0y = etClusEndCap[i] * sin(phiClusEndCap[i]);
	float p1y = etClusEndCap[j] * sin(phiClusEndCap[j]);
	float p0z = eClusEndCap[i] * cos(thetaClusEndCap[i]);
	float p1z = eClusEndCap[j] * cos(thetaClusEndCap[j]);
	float pt_pair = sqrt( (p0x+p1x)*(p0x+p1x) + (p0y+p1y)*(p0y+p1y));
	float m_inv = sqrt ( (eClusEndCap[i] + eClusEndCap[j])*(eClusEndCap[i] + eClusEndCap[j]) - (p0x+p1x)*(p0x+p1x) - (p0y+p1y)*(p0y+p1y) - (p0z+p1z)*(p0z+p1z) );  

	if ( m_inv > seleMinvMaxPi0EndCap_ || m_inv < seleMinvMinPi0EndCap_) continue; 
	
	////try different cut for different regions
	TVector3 pairVect = TVector3((p0x+p1x), (p0y+p1y), (p0z+p1z));
	float etapair = fabs(pairVect.Eta());
	float ptmin = etClusEndCap[i] < etClusEndCap[j] ?  etClusEndCap[i] : etClusEndCap[j]; 
	
	if(etapair <= region1_Pi0EndCap_){
	  if(ptmin < selePtGammaPi0EndCap_region1_ || pt_pair < selePtPi0EndCap_region1_) continue; 
	}else if( etapair <= region2_Pi0EndCap_){
	  if(ptmin < selePtGammaPi0EndCap_region2_ || pt_pair < selePtPi0EndCap_region2_) continue;
	}else{
	  if(ptmin < selePtGammaPi0EndCap_region3_ || pt_pair < selePtPi0EndCap_region3_) continue;
	}
	
	
	//New Loop on cluster to measure isolation:
	vector<int> IsoClus;
	IsoClus.clear();
	float Iso = 0;
	
	for(int k=0 ; k<nClusEndCap ; k++){
	  if(etClusEndCap[k] < ptMinForIsolationEndCap_) continue; 
	  if(k==i || k==j)continue;
	  TVector3 clusVect = TVector3(etClusEndCap[k] * cos(phiClusEndCap[k]), etClusEndCap[k] * sin(phiClusEndCap[k]) , eClusEndCap[k] * cos(thetaClusEndCap[k]) ) ;
	  float dretacl = fabs(etaClusEndCap[k] - pairVect.Eta());
	  float drcl = clusVect.DeltaR(pairVect);
	  
	  if(drcl < selePi0BeltDREndCap_ && dretacl < selePi0BeltDetaEndCap_ ){
	    Iso = Iso + etClusEndCap[k];
	    IsoClus.push_back(k);
	  }
	}
	
	
	if(Iso/pt_pair > selePi0IsoEndCap_) continue; 
	
	
	if( int(RecHitsClusterEndCap[i].size()) < nMinRecHitsSel1stCluster_  || int(RecHitsClusterEndCap[j].size()) < nMinRecHitsSel2ndCluster_) continue; 
	
	
	// 	///Now prescale pi0 selection 
	// 	if(etapair <= region1_Pi0EndCap_){
	// 	  selected_endcapPi0_region1 ++; 
	// 	  if(selected_endcapPi0_region1 % preScale_endcapPi0_region1_ != 0) continue; 
	// 	}else if( etapair <= region2_Pi0EndCap_){
	// 	  selected_endcapPi0_region2 ++; 
	// 	  if(selected_endcapPi0_region2 % preScale_endcapPi0_region2_ != 0) continue; 
	// 	}else{
	// 	  selected_endcapPi0_region3 ++; 
	// 	  if(selected_endcapPi0_region3 % preScale_endcapPi0_region3_ != 0) continue; 
	// 	}
	
	
	
	int indtmp[2]={i,j};
	for(int jj =0; jj<2; jj++){
	  int ind = indtmp[jj];
	  it = find(indClusEndCapSelected.begin(),indClusEndCapSelected.end(),ind);
	  if( it == indClusEndCapSelected.end()){
	    indClusEndCapSelected.push_back(ind);
	    for(unsigned int Rec=0;Rec<RecHitsClusterEndCap[ind].size();Rec++) {
	      selEERecHitCollection->push_back(RecHitsClusterEndCap[ind][Rec]);
	      selectedEEDetIds.push_back(RecHitsClusterEndCap[ind][Rec].id());
	      
	    }
	    
	    if(storeRecHitES_){
	      
	      if( debug_ >= 2){
		cout<<"used_strips for cluster_ee: "<<ind<<" "<< etaClusEndCap[ind]<<" "<<phiClusEndCap[ind]<<" "<<int(used_strips.size())<<endl;
	      }
	      
	      
	      //now call a common function to make ES cluster
	      ///this is to get already done before this cluster
	      ////the used_strips is defined with set, so the order is re-shuffled each time. 
	      std::set<DetId> used_strips_before = used_strips;  
	      makeClusterES(xClusEndCap[ind],yClusEndCap[ind],zClusEndCap[ind],geometry_es,topology_es);
	      std::set<DetId>::const_iterator ites = used_strips.begin();
	      for(; ites != used_strips.end(); ++ ites ){
		ESDetId d1 = ESDetId(*ites);
		if( debug_ >= 2) { cout<<d1<<endl;}
		
		std::set<DetId>::const_iterator ites2 = find(used_strips_before.begin(),used_strips_before.end(),d1);
		if( (ites2 == used_strips_before.end()) ){
		  if( debug_ >= 2) { cout<<d1<<" actually saved"<<endl;}
		  std::map<DetId, EcalRecHit>::iterator itmap = esrechits_map.find(d1);
		  selESRecHitCollection->push_back(itmap->second);
		  
		} 
	      } // loop over used strips
	      
	    } //if (store RecHitsES)
	    
	    
	  } //if cluster was not stored
	  
	} //loop jj
	
	
	if( storeIsoClusRecHitPi0EE_){
	  
	  for(unsigned int iii=0 ; iii<IsoClus.size() ; iii++){   
	    int ind = IsoClus[iii];
	    it = find(indClusEndCapSelected.begin(),indClusEndCapSelected.end(),ind);
	    if( it == indClusEndCapSelected.end()){
	      indClusEndCapSelected.push_back(ind);
	      for(unsigned int Rec3=0;Rec3<RecHitsClusterEndCap[ind].size();Rec3++) {
		selEERecHitCollection->push_back(RecHitsClusterEndCap[ind][Rec3]);
		selectedEEDetIds.push_back(RecHitsClusterEndCap[ind][Rec3].id());
	      }
	      if(storeRecHitES_){
		
		if( debug_ >= 2){
		  cout<<"used_strips for cluster_ee_iso: "<<ind<<" "<< etaClusEndCap[ind]<<" "<<phiClusEndCap[ind]<<" "<<int(used_strips.size())<<endl;
		}
		
		std::set<DetId> used_strips_before = used_strips;  
		makeClusterES(xClusEndCap[ind],yClusEndCap[ind],zClusEndCap[ind],geometry_es,topology_es);
		
		std::set<DetId>::const_iterator ites = used_strips.begin();
		for(; ites != used_strips.end(); ++ ites ){
		  ESDetId d1 = ESDetId(*ites);
		  if( debug_ >= 2) { cout<<d1<<endl;}
		  std::set<DetId>::const_iterator ites2 = find(used_strips_before.begin(),used_strips_before.end(),d1);
		  if( (ites2 == used_strips_before.end())){
		    if( debug_ >= 2) { cout<<d1<<" actually saved"<<endl;}
		    std::map<DetId, EcalRecHit>::iterator itmap = esrechits_map.find(d1);
		    selESRecHitCollection->push_back(itmap->second);
		    
		  }
		  
		} // loop over used strips
		

	      } //if (store RecHitsES)
	      
	    
	    } //if cluster was not stored

	  } // loop over IsoClus

	} //if storeIsoClusRecHitPi0EE_
	
	npi0_se++;
	///	if(npi0_se == MAXPI0S) return false; 
	
      } // End of the "j" loop over Simple Clusters
    } // End of the "i" loop over Simple Clusters
    

  } ///end of selection of pi0->gg endCap
  
  
  ///selection of eta->gg endcap
  if(doSelForEtaEndcap_){

    vector<int> indEtaCand; 
    for(int i=0 ; i<nClusEndCap ; i++){
      for(int j=i+1 ; j<nClusEndCap ; j++){
      
	
	if( s4s9ClusEndCap[i] < seleS4S9GammaEtaEndCap_ || s4s9ClusEndCap[j] < seleS4S9GammaEtaEndCap_
	    || s9s25ClusEndCap[i] < seleS9S25GammaEtaEndCap_ || s9s25ClusEndCap[j] < seleS9S25GammaEtaEndCap_) continue; 
	
	
	float p0x = etClusEndCap[i] * cos(phiClusEndCap[i]);
	float p1x = etClusEndCap[j] * cos(phiClusEndCap[j]);
	float p0y = etClusEndCap[i] * sin(phiClusEndCap[i]);
	float p1y = etClusEndCap[j] * sin(phiClusEndCap[j]);
	float p0z = eClusEndCap[i] * cos(thetaClusEndCap[i]);
	float p1z = eClusEndCap[j] * cos(thetaClusEndCap[j]);
	float pt_pair = sqrt( (p0x+p1x)*(p0x+p1x) + (p0y+p1y)*(p0y+p1y));
	float m_inv =  sqrt ((eClusEndCap[i] + eClusEndCap[j])*(eClusEndCap[i] + eClusEndCap[j]) - (p0x+p1x)*(p0x+p1x) - (p0y+p1y)*(p0y+p1y) - (p0z+p1z)*(p0z+p1z));
	
	
	if ( m_inv > seleMinvMaxEtaEndCap_ || m_inv < seleMinvMinEtaEndCap_) continue; 
	
	////try different cut for different regions
	TVector3 pairVect = TVector3((p0x+p1x), (p0y+p1y), (p0z+p1z));
	float etapair = fabs(pairVect.Eta());
	float ptmin = etClusEndCap[i] < etClusEndCap[j] ?  etClusEndCap[i] : etClusEndCap[j]; 
	
	if(etapair <= region1_EtaEndCap_){
	  if(ptmin < selePtGammaEtaEndCap_region1_ || pt_pair < selePtEtaEndCap_region1_) continue; 
	}else if( etapair <= region2_EtaEndCap_){
	  if(ptmin < selePtGammaEtaEndCap_region2_ || pt_pair < selePtEtaEndCap_region2_) continue;
	}else{
	  if(ptmin < selePtGammaEtaEndCap_region3_ || pt_pair < selePtEtaEndCap_region3_) continue;
	}
		
	
	//New Loop on cluster to measure isolation:
	vector<int> IsoClus;
	IsoClus.clear();
	float Iso = 0;
	for(int k=0 ; k<nClusEndCap ; k++){
	  
	  if(etClusEndCap[k] < ptMinForIsolationEtaEndCap_) continue; 
	  if(k==i || k==j)continue;
	  TVector3 clusVect = TVector3(etClusEndCap[k] * cos(phiClusEndCap[k]), etClusEndCap[k] * sin(phiClusEndCap[k]) , eClusEndCap[k] * cos(thetaClusEndCap[k]));
	  float dretacl = fabs(etaClusEndCap[k] - pairVect.Eta());
	  float drcl = clusVect.DeltaR(pairVect);
	  
	  if(drcl < seleEtaBeltDREndCap_ && dretacl < seleEtaBeltDetaEndCap_ ){
	    Iso = Iso + etClusEndCap[k];
	    IsoClus.push_back(k);
	  }
	}
	
	if(Iso/pt_pair > seleEtaIsoEndCap_) continue; 
	
	if( int(RecHitsClusterEndCap[i].size()) < nMinRecHitsSel1stCluster_  || int(RecHitsClusterEndCap[j].size()) < nMinRecHitsSel2ndCluster_) continue; 
	

	// 	///Now prescale eta selection 
	// 	if(etapair <= region1_EtaEndCap_){
	// 	  selected_endcapEta_region1 ++; 
	// 	  if(selected_endcapEta_region1 % preScale_endcapEta_region1_ != 0) continue; 
	// 	}else if( etapair <= region2_EtaEndCap_){
	// 	  selected_endcapEta_region2 ++; 
	// 	  if(selected_endcapEta_region2 % preScale_endcapEta_region2_ != 0) continue; 
	// 	}else{
	// 	  selected_endcapEta_region3 ++; 
	// 	  if(selected_endcapEta_region3 % preScale_endcapEta_region3_ != 0) continue; 
	// 	}
	
	
	
	
	int indtmp[2]={i,j};
	for(int jj =0; jj<2; jj++){
	  int ind = indtmp[jj];
	  
	  ///eta candidates
	  it = find(indEtaCand.begin(),indEtaCand.end(),ind);
	  if(it == indEtaCand.end()){
	    indEtaCand.push_back(ind);
	  }
		
	  it = find(indClusEndCapSelected.begin(),indClusEndCapSelected.end(),ind);
	  if( it == indClusEndCapSelected.end()){
	    indClusEndCapSelected.push_back(ind);
	    for(unsigned int Rec=0;Rec<RecHitsClusterEndCap[ind].size();Rec++) {
	      selEERecHitCollection->push_back(RecHitsClusterEndCap[ind][Rec]);
	      selectedEEDetIds.push_back(RecHitsClusterEndCap[ind][Rec].id());
	    }
	    
	    if(storeRecHitES_){
	      if( debug_ >= 2){
		cout<<"used_strips for cluster_ee_eta: "<<ind<<" "<< etaClusEndCap[ind]<<" "<<phiClusEndCap[ind]<<" "<<int(used_strips.size())<<endl;
	      }
	      
	      std::set<DetId> used_strips_before = used_strips;  
	      makeClusterES(xClusEndCap[ind],yClusEndCap[ind],zClusEndCap[ind],geometry_es,topology_es);
	      std::set<DetId>::const_iterator ites = used_strips.begin();
	      for(; ites != used_strips.end(); ++ ites ){
		ESDetId d1 = ESDetId(*ites);
		if( debug_ >= 2) { cout<<d1<<endl;}
		std::set<DetId>::const_iterator ites2 = find(used_strips_before.begin(),used_strips_before.end(),d1);
                if( (ites2 == used_strips_before.end())){
		  if( debug_ >= 2) { cout<<d1<<" actually saved"<<endl;}
		  std::map<DetId, EcalRecHit>::iterator itmap = esrechits_map.find(d1);
		  selESRecHitCollection->push_back(itmap->second);
		  
		} 
	      } // loop over used strips
	      
	    } //if (store RecHitsES)
	    
	    
	  } //if cluster was not stored
	  

	} // loop jj
	
	
	if( storeIsoClusRecHitEtaEE_){
	  
	  for(unsigned int iii=0 ; iii<IsoClus.size() ; iii++){   
	    int ind = IsoClus[iii];
	    
	    if(store5x5IsoClusRecHitEtaEE_){
	      ///eta candidates IsoClus.
	      it = find(indEtaCand.begin(),indEtaCand.end(),ind);
	      if(it == indEtaCand.end()){
		indEtaCand.push_back(ind);
	      }
	    }
	    
	    it = find(indClusEndCapSelected.begin(),indClusEndCapSelected.end(),ind);
	    if( it == indClusEndCapSelected.end()){
	      indClusEndCapSelected.push_back(ind);
	      for(unsigned int Rec3=0;Rec3<RecHitsClusterEndCap[ind].size();Rec3++) {
		selEERecHitCollection->push_back(RecHitsClusterEndCap[ind][Rec3]);
		selectedEEDetIds.push_back(RecHitsClusterEndCap[ind][Rec3].id());

	      }
	      if(storeRecHitES_){
		if( debug_ >= 2){
		  cout<<"used_strips for cluster_eeiso_eta: "<<ind<<" "<< etaClusEndCap[ind]<<" "<<phiClusEndCap[ind]<<" "<<int(used_strips.size())<<endl;
		}
		
		std::set<DetId> used_strips_before = used_strips;  
		makeClusterES(xClusEndCap[ind],yClusEndCap[ind],zClusEndCap[ind],geometry_es,topology_es);
		
		std::set<DetId>::const_iterator ites = used_strips.begin();
		for(; ites != used_strips.end(); ++ ites ){
		  ESDetId d1 = ESDetId(*ites);
		  if( debug_ >= 2) { cout<<d1<<endl;}
		  std::set<DetId>::const_iterator ites2 = find(used_strips_before.begin(),used_strips_before.end(),d1);
		  if( (ites2 == used_strips_before.end())){
		    if( debug_ >= 2) { cout<<d1<<" actually saved"<<endl;}
		    std::map<DetId, EcalRecHit>::iterator itmap = esrechits_map.find(d1);
		    selESRecHitCollection->push_back(itmap->second);
		    
		  } 
		} // loop over used strips
		
	      } //if (store RecHitsES)
	    
	    } //if cluster was not stored
	    
	    
          } // loop over iii IsoClus

        } //if storeIsoClusRecHitEtaEE_

	
      } // End of the "j" loop over Simple Clusters
    } // End of the "i" loop over Simple Clusters
    
    
    
    if(store5x5RecHitEtaEE_){
      ///for selected eta->gg candidates save 5x5 rechits also
      for(int j=0; j<int(indEtaCand.size());j++){
	int ind = indEtaCand[j];
	for(unsigned int Rec3=0;Rec3<RecHitsCluster5x5EndCap[ind].size();Rec3++) {
	  
	  DetId det = RecHitsCluster5x5EndCap[ind][Rec3].id();
	  std::vector<DetId>::iterator itdet = find(selectedEEDetIds.begin(),selectedEEDetIds.end(),det);
	  if(itdet == selectedEEDetIds.end()){
	    selectedEEDetIds.push_back(det);
	    selEERecHitCollection->push_back(RecHitsCluster5x5EndCap[ind][Rec3]);
	    
	  }
	  
	}
      }
    }
    
  }///end of selections eta->gg endcap
  
  
  delete topology_es;
  
  
  
  ////==============End of endcap ===============///
  
  if(debug_>=1) std::cout<<" endcap_output_size: "<<iEvent.id().run()<<" "<<iEvent.id().event()<<" "<<selEERecHitCollection->size()<<" "<<selESRecHitCollection->size()<<std::endl;
  
  
  //Put selected information in the event
  int collsize = int(selEBRecHitCollection->size());
  int collsizeEndCap = int(selEERecHitCollection->size());
  
  
  ///no rechits selected.
  if( collsize < 2 && collsizeEndCap <2) return false; 
  
  ///too many rechits.
  ///  if(collsize + collsizeEndCap > seleNRHMax_ ) return false; 
  
  
  

  ////Now put into events selected rechits.
  if(doBarrel){
    iEvent.put( selEBRecHitCollection, BarrelHits_);
  }  
  
  if(doEndcap){
    iEvent.put( selEERecHitCollection, EndcapHits_);
    
    if(storeRecHitES_){
      iEvent.put( selESRecHitCollection, ESHits_);
    }
  }
  
  return true; 
  
  
  
  
  
}


void HLTPi0RecHitsFilter::makeClusterES(float x, float y, float z,const CaloSubdetectorGeometry*& geometry_es,
					CaloSubdetectorTopology*& topology_es
					){
  
  
  ///get assosicated ES clusters of this endcap cluster
  const GlobalPoint point(x,y,z);
  DetId tmp1 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_es))->getClosestCellInPlane(point, 1);
  DetId tmp2 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_es))->getClosestCellInPlane(point, 2);
  ESDetId strip1 = (tmp1 == DetId(0)) ? ESDetId(0) : ESDetId(tmp1);
  ESDetId strip2 = (tmp2 == DetId(0)) ? ESDetId(0) : ESDetId(tmp2);     
  
  // Get ES clusters (found by the PreshSeeded algorithm) associated with a given EE cluster.           
  for (int i2=0; i2<preshNclust_; i2++) {
    reco::PreshowerCluster cl1 = presh_algo->makeOneCluster(strip1,&used_strips,&esrechits_map,geometry_es,topology_es);   
    reco::PreshowerCluster cl2 = presh_algo->makeOneCluster(strip2,&used_strips,&esrechits_map,geometry_es,topology_es); 
  } // end of cycle over ES clusters
    
  
}


/////FED list this is obsolete 
// std::vector<int> HLTPi0RecHitsFilter::ListOfFEDS(double etaLow, double etaHigh, double phiLow, 
// 					 double phiHigh, double etamargin, double phimargin)
// {

// 	std::vector<int> FEDs;

// 	if (phimargin > Geom::pi()) phimargin =  Geom::pi() ;

	
// 	if (debug_>=2) std::cout << " etaLow etaHigh phiLow phiHigh " << etaLow << " " << 
// 			etaHigh << " " << phiLow << " " << phiHigh << std::endl;

//         etaLow -= etamargin;
//         etaHigh += etamargin;
//         double phiMinus = phiLow - phimargin;
//         double phiPlus = phiHigh + phimargin;

//         bool all = false;
//         double dd = fabs(phiPlus-phiMinus);
// 	if (debug_>=2) std::cout << " dd = " << dd << std::endl;
//         if (dd > 2.*Geom::pi() ) all = true;

//         while (phiPlus > Geom::pi()) { phiPlus -= 2.*Geom::pi() ; }
//         while (phiMinus < 0) { phiMinus += 2.*Geom::pi() ; }
//         if ( phiMinus > Geom::pi()) phiMinus -= 2.*Geom::pi() ;

//         double dphi = phiPlus - phiMinus;
//         if (dphi < 0) dphi += 2.*Geom::pi() ;
// 	if (debug_>=2) std::cout << "dphi = " << dphi << std::endl;
//         if (dphi > Geom::pi()) {
//                 int fed_low1 = TheMapping -> GetFED(etaLow,phiMinus*180./Geom::pi());
//                 int fed_low2 = TheMapping -> GetFED(etaLow,phiPlus*180./Geom::pi());
// 		if (debug_>=2) std::cout << "fed_low1 fed_low2 " << fed_low1 << " " << fed_low2 << std::endl;
//                 if (fed_low1 == fed_low2) all = true;
//                 int fed_hi1 = TheMapping -> GetFED(etaHigh,phiMinus*180./Geom::pi());
//                 int fed_hi2 = TheMapping -> GetFED(etaHigh,phiPlus*180./Geom::pi());
// 		if (debug_>=2) std::cout << "fed_hi1 fed_hi2 " << fed_hi1 << " " << fed_hi2 << std::endl;
//                 if (fed_hi1 == fed_hi2) all = true;
//         }

// 	if (all) {
// 		if (debug_>=2) std::cout << " unpack everything in phi ! " << std::endl;
// 		phiMinus = -20 * Geom::pi() / 180.;  // -20 deg
// 		phiPlus = -40 * Geom::pi() / 180.;  // -20 deg
// 	}

//         if (debug_>=2) std::cout << " with margins : " << etaLow << " " << etaHigh << " " << 
// 			phiMinus << " " << phiPlus << std::endl;


//         const EcalEtaPhiRegion ecalregion(etaLow,etaHigh,phiMinus,phiPlus);

//         FEDs = TheMapping -> GetListofFEDs(ecalregion);

// /*
// 	if (debug_) {
//            int nn = (int)FEDs.size();
//            for (int ii=0; ii < nn; ii++) {
//                    std::cout << "unpack fed " << FEDs[ii] << std::endl;
//            }
//    	   }
// */

//         return FEDs;

// }


////already existing , int EcalElectronicsMapping::DCCid(const EBDetId& id)
///obsolete
// int HLTPi0RecHitsFilter::convertSmToFedNumbBarrel(int ieta, int smId){
    
//   if( ieta<=-1) return smId - 9; 
//   else return smId + 27; 
  
  
// }


void HLTPi0RecHitsFilter::convxtalid(int &nphi,int &neta)
{
  // Barrel only
  // Output nphi 0...359; neta 0...84; nside=+1 (for eta>0), or 0 (for eta<0).
  // neta will be [-85,-1] , or [0,84], the minus sign indicates the z<0 side.
  
  if(neta > 0) neta -= 1;
  if(nphi > 359) nphi=nphi-360;
  
  // final check
  if(nphi >359 || nphi <0 || neta< -85 || neta > 84)
    {
      std::cout <<" unexpected fatal error in HLTPi0RecHitsFilter::convxtalid "<<  nphi <<  " " << neta <<  " " <<std::endl;
      //exit(1);
    }
} //end of convxtalid




int HLTPi0RecHitsFilter::diff_neta_s(int neta1, int neta2){
  int mdiff;
  mdiff=(neta1-neta2);
  return mdiff;
}

// Calculate the distance in xtals taking into account the periodicity of the Barrel
int HLTPi0RecHitsFilter::diff_nphi_s(int nphi1,int nphi2) {
   int mdiff;
   if(std::abs(nphi1-nphi2) < (360-std::abs(nphi1-nphi2))) {
     mdiff=nphi1-nphi2;
   }
   else {
   mdiff=360-std::abs(nphi1-nphi2);
   if(nphi1>nphi2) mdiff=-mdiff;
   }
   return mdiff;
}

