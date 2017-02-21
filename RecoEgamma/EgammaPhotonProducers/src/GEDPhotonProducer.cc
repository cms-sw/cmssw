#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "CommonTools/Utils/interface/StringToEnumValue.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"


#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "RecoEgamma/EgammaPhotonProducers/interface/GEDPhotonProducer.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h" 
#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterCrackCorrection.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"

namespace {
  inline double ptFast( const double energy, 
			const math::XYZPoint& position,
			const math::XYZPoint& origin ) {
    const auto v = position - origin;
    return energy*std::sqrt(v.perp2()/v.mag2());
  }
}

GEDPhotonProducer::GEDPhotonProducer(const edm::ParameterSet& config) : 

  conf_(config)
{

  // use configuration file to setup input/output collection names
  //
  photonProducer_       = conf_.getParameter<edm::InputTag>("photonProducer");
  reconstructionStep_   = conf_.getParameter<std::string>("reconstructionStep");

  if (  reconstructionStep_ == "final" ) {
    photonProducerT_   = 
      consumes<reco::PhotonCollection>(photonProducer_);
    pfCandidates_      = 
      consumes<reco::PFCandidateCollection>(conf_.getParameter<edm::InputTag>("pfCandidates"));

    phoChargedIsolationToken_CITK      = 
      consumes<edm::ValueMap<float>>(conf_.getParameter<edm::InputTag>("chargedHadronIsolation"));
    phoNeutralHadronIsolationToken_CITK      = 
      consumes<edm::ValueMap<float>>(conf_.getParameter<edm::InputTag>("neutralHadronIsolation"));
    phoPhotonIsolationToken_CITK      = 
      consumes<edm::ValueMap<float>>(conf_.getParameter<edm::InputTag>("photonIsolation"));   

  } else {

    photonCoreProducerT_   = 
      consumes<reco::PhotonCoreCollection>(photonProducer_);

  }

  pfEgammaCandidates_      = 
    consumes<reco::PFCandidateCollection>(conf_.getParameter<edm::InputTag>("pfEgammaCandidates"));
  barrelEcalHits_   = 
    consumes<EcalRecHitCollection>(conf_.getParameter<edm::InputTag>("barrelEcalHits"));
  endcapEcalHits_   = 
    consumes<EcalRecHitCollection>(conf_.getParameter<edm::InputTag>("endcapEcalHits"));
  preshowerHits_   = 
    consumes<EcalRecHitCollection>(conf_.getParameter<edm::InputTag>("preshowerHits"));
  vertexProducer_   = 
    consumes<reco::VertexCollection>(conf_.getParameter<edm::InputTag>("primaryVertexProducer"));

  hcalTowers_ = 
    consumes<CaloTowerCollection>(conf_.getParameter<edm::InputTag>("hcalTowers"));
  //
  photonCollection_     = conf_.getParameter<std::string>("outputPhotonCollection");
  hOverEConeSize_   = conf_.getParameter<double>("hOverEConeSize");
  highEt_        = conf_.getParameter<double>("highEt");
  // R9 value to decide converted/unconverted
  minR9Barrel_        = conf_.getParameter<double>("minR9Barrel");
  minR9Endcap_        = conf_.getParameter<double>("minR9Endcap");
  usePrimaryVertex_   = conf_.getParameter<bool>("usePrimaryVertex");
  runMIPTagger_       = conf_.getParameter<bool>("runMIPTagger");

  candidateP4type_ = config.getParameter<std::string>("candidateP4type") ;
  valueMapPFCandPhoton_ = config.getParameter<std::string>("valueMapPhotons");


  edm::ParameterSet posCalcParameters = 
    config.getParameter<edm::ParameterSet>("posCalcParameters");
  posCalculator_ = PositionCalc(posCalcParameters);


  //AA
  //Flags and Severities to be excluded from photon calculations
  const std::vector<std::string> flagnamesEB = 
    config.getParameter<std::vector<std::string> >("RecHitFlagToBeExcludedEB");

  const std::vector<std::string> flagnamesEE =
    config.getParameter<std::vector<std::string> >("RecHitFlagToBeExcludedEE");

  flagsexclEB_= 
    StringToEnumValue<EcalRecHit::Flags>(flagnamesEB);

  flagsexclEE_=
    StringToEnumValue<EcalRecHit::Flags>(flagnamesEE);

  const std::vector<std::string> severitynamesEB = 
    config.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcludedEB");

  severitiesexclEB_= 
    StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEB);

  const std::vector<std::string> severitynamesEE = 
    config.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcludedEE");

  severitiesexclEE_= 
    StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEE);

  thePhotonEnergyCorrector_ = 
    new PhotonEnergyCorrector(conf_, consumesCollector());
  if( conf_.existsAs<edm::ParameterSet>("regressionConfig") ) {
    auto sumes = consumesCollector();
    thePhotonEnergyCorrector_->gedRegression()->setConsumes(sumes);
  }

  //AA

  //

  // Parameters for the position calculation:
  //  std::map<std::string,double> providedParameters;
  // providedParameters.insert(std::make_pair("LogWeighted",conf_.getParameter<bool>("posCalc_logweight")));
  //providedParameters.insert(std::make_pair("T0_barl",conf_.getParameter<double>("posCalc_t0_barl")));
  //providedParameters.insert(std::make_pair("T0_endc",conf_.getParameter<double>("posCalc_t0_endc")));
  //providedParameters.insert(std::make_pair("T0_endcPresh",conf_.getParameter<double>("posCalc_t0_endcPresh")));
  //providedParameters.insert(std::make_pair("W0",conf_.getParameter<double>("posCalc_w0")));
  //providedParameters.insert(std::make_pair("X0",conf_.getParameter<double>("posCalc_x0")));
  //posCalculator_ = PositionCalc(providedParameters);
  // cut values for pre-selection
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("minSCEtBarrel")); 
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("maxHoverEBarrel")); 
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("ecalRecHitSumEtOffsetBarrel")); 
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("ecalRecHitSumEtSlopeBarrel")); 
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("hcalTowerSumEtOffsetBarrel"));
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("hcalTowerSumEtSlopeBarrel"));
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("nTrackSolidConeBarrel"));
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("nTrackHollowConeBarrel"));     
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("trackPtSumSolidConeBarrel"));     
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("trackPtSumHollowConeBarrel"));     
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("sigmaIetaIetaCutBarrel"));     
  //  
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("minSCEtEndcap")); 
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("maxHoverEEndcap")); 
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("ecalRecHitSumEtOffsetEndcap")); 
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("ecalRecHitSumEtSlopeEndcap")); 
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("hcalTowerSumEtOffsetEndcap"));
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("hcalTowerSumEtSlopeEndcap"));
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("nTrackSolidConeEndcap"));
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("nTrackHollowConeEndcap"));     
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("trackPtSumSolidConeEndcap"));     
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("trackPtSumHollowConeEndcap"));     
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("sigmaIetaIetaCutEndcap"));     
  //

  //moved from beginRun to here, I dont see how this could cause harm as its just reading in the exactly same parameters each run
  if ( reconstructionStep_ != "final"){
    thePhotonIsolationCalculator_ = new PhotonIsolationCalculator();
    edm::ParameterSet isolationSumsCalculatorSet = conf_.getParameter<edm::ParameterSet>("isolationSumsCalculatorSet"); 
    thePhotonIsolationCalculator_->setup(isolationSumsCalculatorSet, flagsexclEB_, flagsexclEE_, severitiesexclEB_, severitiesexclEE_,consumesCollector());
    thePhotonMIPHaloTagger_ = new PhotonMIPHaloTagger();
    edm::ParameterSet mipVariableSet = conf_.getParameter<edm::ParameterSet>("mipVariableSet"); 
    thePhotonMIPHaloTagger_->setup(mipVariableSet,consumesCollector());
    
  }else{
    thePhotonIsolationCalculator_=0;
    thePhotonMIPHaloTagger_=0;
  }
  // Register the product
  produces< reco::PhotonCollection >(photonCollection_);
  produces< edm::ValueMap<reco::PhotonRef> > (valueMapPFCandPhoton_);


}

GEDPhotonProducer::~GEDPhotonProducer() 
{
  delete thePhotonEnergyCorrector_;
  delete thePhotonIsolationCalculator_;
  delete thePhotonMIPHaloTagger_;
 //delete energyCorrectionF;
}



void  GEDPhotonProducer::beginRun (edm::Run const& r, edm::EventSetup const & theEventSetup) {

 if ( reconstructionStep_ != "final" ) { 
    thePhotonEnergyCorrector_ -> init(theEventSetup); 
  }

}

void  GEDPhotonProducer::endRun (edm::Run const& r, edm::EventSetup const & theEventSetup) {
}


void GEDPhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  using namespace edm;
  //  nEvt_++;
 
  reco::PhotonCollection outputPhotonCollection;
  auto outputPhotonCollection_p = std::make_unique<reco::PhotonCollection>();
  edm::ValueMap<reco::PhotonRef> pfEGCandToPhotonMap;


  // Get the PhotonCore collection
  bool validPhotonCoreHandle=false;
  Handle<reco::PhotonCoreCollection> photonCoreHandle;
  bool validPhotonHandle= false;
  Handle<reco::PhotonCollection> photonHandle;
  //value maps for isolation
  edm::Handle<edm::ValueMap<float> > phoChargedIsolationMap_CITK;
  edm::Handle<edm::ValueMap<float> > phoNeutralHadronIsolationMap_CITK;
  edm::Handle<edm::ValueMap<float> > phoPhotonIsolationMap_CITK;

  if ( reconstructionStep_ == "final" ) { 
    theEvent.getByToken(photonProducerT_,photonHandle);
    //get isolation objects
    theEvent.getByToken(phoChargedIsolationToken_CITK,phoChargedIsolationMap_CITK);
    theEvent.getByToken(phoNeutralHadronIsolationToken_CITK,phoNeutralHadronIsolationMap_CITK);
    theEvent.getByToken(phoPhotonIsolationToken_CITK,phoPhotonIsolationMap_CITK);
    if ( photonHandle.isValid()) {
      validPhotonHandle=true;  
    } else {
      throw cms::Exception("GEDPhotonProducer") << "Error! Can't get the product " <<   photonProducer_.label() << "\n";
    }
  } else {
    
    theEvent.getByToken(photonCoreProducerT_,photonCoreHandle);
    if (photonCoreHandle.isValid()) {
      validPhotonCoreHandle=true;
    } else {
      throw cms::Exception("GEDPhotonProducer") 
	<< "Error! Can't get the photonCoreProducer" <<  photonProducer_.label() << "\n";
    } 
  }

  // Get EcalRecHits
  bool validEcalRecHits=true;
  Handle<EcalRecHitCollection> barrelHitHandle;
  const EcalRecHitCollection dummyEB;
  theEvent.getByToken(barrelEcalHits_, barrelHitHandle);
  if (!barrelHitHandle.isValid()) {
    throw cms::Exception("GEDPhotonProducer") 
      << "Error! Can't get the barrelEcalHits";
  }
  const EcalRecHitCollection& barrelRecHits(validEcalRecHits ? *(barrelHitHandle.product()) : dummyEB);
  
  Handle<EcalRecHitCollection> endcapHitHandle;
  theEvent.getByToken(endcapEcalHits_, endcapHitHandle);
  const EcalRecHitCollection dummyEE;
  if (!endcapHitHandle.isValid()) {
    throw cms::Exception("GEDPhotonProducer") 
      << "Error! Can't get the endcapEcalHits";
  }
  const EcalRecHitCollection& endcapRecHits(validEcalRecHits ? *(endcapHitHandle.product()) : dummyEE);

  bool validPreshowerRecHits=true;
  Handle<EcalRecHitCollection> preshowerHitHandle;
  theEvent.getByToken(preshowerHits_, preshowerHitHandle);
  EcalRecHitCollection preshowerRecHits;
  if (!preshowerHitHandle.isValid()) {
    throw cms::Exception("GEDPhotonProducer") 
      << "Error! Can't get the preshowerEcalHits";
  }
  if( validPreshowerRecHits ) preshowerRecHits = *(preshowerHitHandle.product());



  Handle<reco::PFCandidateCollection> pfEGCandidateHandle;
  // Get the  PF refined cluster  collection
  theEvent.getByToken(pfEgammaCandidates_,pfEGCandidateHandle);
  if (!pfEGCandidateHandle.isValid()) {
    throw cms::Exception("GEDPhotonProducer") 
      << "Error! Can't get the pfEgammaCandidates";
  }
  
  Handle<reco::PFCandidateCollection> pfCandidateHandle;

  if ( reconstructionStep_ == "final" ) {  
    // Get the  PF candidates collection
    theEvent.getByToken(pfCandidates_,pfCandidateHandle);
    if (!pfCandidateHandle.isValid()) {
      throw cms::Exception("GEDPhotonProducer") 
	<< "Error! Can't get the pfCandidates";
    }
  } 

  //AA
  //Get the severity level object
  edm::ESHandle<EcalSeverityLevelAlgo> sevLv;
  theEventSetup.get<EcalSeverityLevelAlgoRcd>().get(sevLv);
  //


// get Hcal towers collection 
  Handle<CaloTowerCollection> hcalTowersHandle;
  theEvent.getByToken(hcalTowers_, hcalTowersHandle);


  // get the geometry from the event setup:
  theEventSetup.get<CaloGeometryRecord>().get(theCaloGeom_);

  //
  // update energy correction function
  //  energyCorrectionF->init(theEventSetup);  

  edm::ESHandle<CaloTopology> pTopology;
  theEventSetup.get<CaloTopologyRecord>().get(theCaloTopo_);
  const CaloTopology *topology = theCaloTopo_.product();

  // Get the primary event vertex
  Handle<reco::VertexCollection> vertexHandle;
  const reco::VertexCollection dummyVC;
  bool validVertex=true;
  if ( usePrimaryVertex_ ) {
    theEvent.getByToken(vertexProducer_, vertexHandle);
    if (!vertexHandle.isValid()) {
      throw cms::Exception("GEDPhotonProducer") 
	<< "Error! Can't get the product primary Vertex Collection";
    }
  }
  const reco::VertexCollection& vertexCollection(usePrimaryVertex_ && validVertex ? *(vertexHandle.product()) : dummyVC);

  //  math::XYZPoint vtx(0.,0.,0.);
  //if (vertexCollection.size()>0) vtx = vertexCollection.begin()->position();

  // get the regression calculator ready
  thePhotonEnergyCorrector_->init(theEventSetup);
  if( thePhotonEnergyCorrector_->gedRegression() ) {
    thePhotonEnergyCorrector_->gedRegression()->setEvent(theEvent);
    thePhotonEnergyCorrector_->gedRegression()->setEventContent(theEventSetup);
  }


  int iSC=0; // index in photon collection
  // Loop over barrel and endcap SC collections and fill the  photon collection
  if ( validPhotonCoreHandle) 
    fillPhotonCollection(theEvent,
			 theEventSetup,
			 photonCoreHandle,
			 topology,
			 &barrelRecHits,
			 &endcapRecHits,
                         &preshowerRecHits,
			 hcalTowersHandle,
			 //vtx,
			 vertexCollection,
			 outputPhotonCollection,
			 iSC);

  iSC=0;
  if ( validPhotonHandle &&  reconstructionStep_ == "final" )
    fillPhotonCollection(theEvent,
			 theEventSetup,
			 photonHandle,
			 pfCandidateHandle,
			 pfEGCandidateHandle,
			 pfEGCandToPhotonMap,
			 vertexHandle,
			 outputPhotonCollection,
			 iSC,
       phoChargedIsolationMap_CITK,
       phoNeutralHadronIsolationMap_CITK,
       phoPhotonIsolationMap_CITK);



  // put the product in the event
  edm::LogInfo("GEDPhotonProducer") << " Put in the event " << iSC << " Photon Candidates \n";
  outputPhotonCollection_p->assign(outputPhotonCollection.begin(),outputPhotonCollection.end());
  const edm::OrphanHandle<reco::PhotonCollection> photonOrphHandle = theEvent.put(std::move(outputPhotonCollection_p), photonCollection_);


  if ( reconstructionStep_ != "final" ) { 
    //// Define the value map which associate to each  Egamma-unbiassaed candidate (key-ref) the corresponding PhotonRef 
    auto pfEGCandToPhotonMap_p = std::make_unique<edm::ValueMap<reco::PhotonRef>>();
    edm::ValueMap<reco::PhotonRef>::Filler filler(*pfEGCandToPhotonMap_p);
    unsigned nObj = pfEGCandidateHandle->size();
    std::vector<reco::PhotonRef> values(nObj);
    //// Fill the value map which associate to each Photon (key) the corresponding Egamma-unbiassaed candidate (value-ref) 
    for(unsigned int lCand=0; lCand < nObj; lCand++) {
      reco::PFCandidateRef pfCandRef (reco::PFCandidateRef(pfEGCandidateHandle,lCand));
      reco::SuperClusterRef pfScRef = pfCandRef -> superClusterRef(); 
      
      for(unsigned int lSC=0; lSC < photonOrphHandle->size(); lSC++) {
	reco::PhotonRef photonRef(reco::PhotonRef(photonOrphHandle, lSC));
	reco::SuperClusterRef scRef=photonRef->superCluster();
	if ( pfScRef != scRef ) continue;
	values[lCand] = photonRef; 
      }
    }
    
    
    filler.insert(pfEGCandidateHandle,values.begin(),values.end());
    filler.fill(); 
    theEvent.put(std::move(pfEGCandToPhotonMap_p),valueMapPFCandPhoton_);


  }
    
    
    



}

void GEDPhotonProducer::fillPhotonCollection(edm::Event& evt,
					     edm::EventSetup const & es,
					     const edm::Handle<reco::PhotonCoreCollection> & photonCoreHandle,
					     const CaloTopology* topology,
					     const EcalRecHitCollection* ecalBarrelHits,
					     const EcalRecHitCollection* ecalEndcapHits,
                                             const EcalRecHitCollection* preshowerHits,
					     const edm::Handle<CaloTowerCollection> & hcalTowersHandle, 
					     const reco::VertexCollection & vertexCollection,
					     reco::PhotonCollection & outputPhotonCollection, int& iSC) {
  
  
  const CaloGeometry* geometry = theCaloGeom_.product();
  const EcalRecHitCollection* hits = nullptr ;
  std::vector<double> preselCutValues;
  std::vector<int> flags_, severitiesexcl_;

  for(unsigned int lSC=0; lSC < photonCoreHandle->size(); lSC++) {

    reco::PhotonCoreRef coreRef(reco::PhotonCoreRef(photonCoreHandle, lSC));
    reco::SuperClusterRef parentSCRef = coreRef->parentSuperCluster();
    reco::SuperClusterRef scRef=coreRef->superCluster();

  
  
    //    const reco::SuperCluster* pClus=&(*scRef);
    iSC++;
    
    int thedet = scRef->seed()->hitsAndFractions()[0].first.det();
    int subdet = scRef->seed()->hitsAndFractions()[0].first.subdetId();
    if (subdet==EcalBarrel) { 
      preselCutValues = preselCutValuesBarrel_;
      hits = ecalBarrelHits;
      flags_ = flagsexclEB_;
      severitiesexcl_ = severitiesexclEB_;
    } else if (subdet==EcalEndcap)  { 
      preselCutValues = preselCutValuesEndcap_;
      hits = ecalEndcapHits;
      flags_ = flagsexclEE_;
      severitiesexcl_ = severitiesexclEE_;
    } else if ( thedet == DetId::Forward )  {
      preselCutValues = preselCutValuesEndcap_;
      hits = nullptr;
      flags_ = flagsexclEE_;
      severitiesexcl_ = severitiesexclEE_;
    } else {
      edm::LogWarning("")<<"GEDPhotonProducer: do not know if it is a barrel or endcap SuperCluster" << thedet << ' ' << subdet; 
    }

    
    

    // SC energy preselection
    if (parentSCRef.isNonnull() &&
	ptFast(parentSCRef->energy(),parentSCRef->position(),math::XYZPoint(0,0,0)) <= preselCutValues[0] ) continue;
    // calculate HoE    

    const CaloTowerCollection* hcalTowersColl = hcalTowersHandle.product();
    EgammaTowerIsolation towerIso1(hOverEConeSize_,0.,0.,1,hcalTowersColl) ;  
    EgammaTowerIsolation towerIso2(hOverEConeSize_,0.,0.,2,hcalTowersColl) ;  
    double HoE1=towerIso1.getTowerESum(&(*scRef))/scRef->energy();
    double HoE2=towerIso2.getTowerESum(&(*scRef))/scRef->energy(); 
    
    EgammaHadTower towerIsoBehindClus(es); 
    towerIsoBehindClus.setTowerCollection(hcalTowersHandle.product());
    std::vector<CaloTowerDetId> TowersBehindClus =  towerIsoBehindClus.towersOf(*scRef);
    float hcalDepth1OverEcalBc = towerIsoBehindClus.getDepth1HcalESum(TowersBehindClus)/scRef->energy();
    float hcalDepth2OverEcalBc = towerIsoBehindClus.getDepth2HcalESum(TowersBehindClus)/scRef->energy();
    //    std::cout << " GEDPhotonProducer calculation of HoE with towers in a cone " << HoE1  << "  " << HoE2 << std::endl;
    //std::cout << " GEDPhotonProducer calcualtion of HoE with towers behind the BCs " << hcalDepth1OverEcalBc  << "  " << hcalDepth2OverEcalBc << std::endl;

    float maxXtal = ( hits != nullptr ? EcalClusterTools::eMax( *(scRef->seed()), &(*hits) ) : 0.f );
    //AA
    //Change these to consider severity level of hits
    float e1x5    =   ( hits != nullptr ? EcalClusterTools::e1x5(  *(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    float e2x5    =   ( hits != nullptr ? EcalClusterTools::e2x5Max(  *(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    float e3x3    =   ( hits != nullptr ? EcalClusterTools::e3x3(  *(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    float e5x5    =   ( hits != nullptr ? EcalClusterTools::e5x5( *(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    std::vector<float> cov =  ( hits != nullptr ? EcalClusterTools::covariances( *(scRef->seed()), &(*hits), &(*topology), geometry) : std::vector<float>( {0.f,0.f,0.f} ) );
    std::vector<float> locCov =  ( hits != nullptr ? EcalClusterTools::localCovariances( *(scRef->seed()), &(*hits), &(*topology)) : std::vector<float>( {0.f,0.f,0.f} ) );
      
    float sigmaEtaEta = sqrt(cov[0]);
    float sigmaIetaIeta = sqrt(locCov[0]);
    
    float full5x5_maxXtal =   ( hits != nullptr ? noZS::EcalClusterTools::eMax( *(scRef->seed()), &(*hits) ) : 0.f );
    //AA
    //Change these to consider severity level of hits
    float full5x5_e1x5    =   ( hits != nullptr ? noZS::EcalClusterTools::e1x5(  *(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    float full5x5_e2x5    =   ( hits != nullptr ? noZS::EcalClusterTools::e2x5Max(  *(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    float full5x5_e3x3    =   ( hits != nullptr ? noZS::EcalClusterTools::e3x3(  *(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    float full5x5_e5x5    =   ( hits != nullptr ? noZS::EcalClusterTools::e5x5( *(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    std::vector<float> full5x5_cov =  ( hits != nullptr ? noZS::EcalClusterTools::covariances( *(scRef->seed()), &(*hits), &(*topology), geometry) : std::vector<float>( {0.f,0.f,0.f} ) );
    std::vector<float> full5x5_locCov =  ( hits != nullptr ? noZS::EcalClusterTools::localCovariances( *(scRef->seed()), &(*hits), &(*topology)) : std::vector<float>( {0.f,0.f,0.f} ) );
      
    float full5x5_sigmaEtaEta = sqrt(full5x5_cov[0]);
    float full5x5_sigmaIetaIeta = sqrt(full5x5_locCov[0]);    
    
    // compute position of ECAL shower
    math::XYZPoint caloPosition = scRef->position();
    

    //// energy determination -- Default to create the candidate. Afterwards corrections are applied
    double photonEnergy=1.;
    math::XYZPoint vtx(0.,0.,0.);
    if (vertexCollection.size()>0) vtx = vertexCollection.begin()->position();
    // compute momentum vector of photon from primary vertex and cluster position
    math::XYZVector direction = caloPosition - vtx;
    //math::XYZVector momentum = direction.unit() * photonEnergy ;
    math::XYZVector momentum = direction.unit() ;

    // Create dummy candidate with unit momentum and zero energy to allow setting of all variables. The energy is set for last.
    math::XYZTLorentzVectorD p4(momentum.x(), momentum.y(), momentum.z(), photonEnergy );
    reco::Photon newCandidate(p4, caloPosition, coreRef, vtx);

    //std::cout << " standard p4 before " << newCandidate.p4() << " energy " << newCandidate.energy() <<  std::endl;
    //std::cout << " type " <<newCandidate.getCandidateP4type() <<  " standard p4 after " << newCandidate.p4() << " energy " << newCandidate.energy() << std::endl;

    // Calculate fiducial flags and isolation variable. Blocked are filled from the isolationCalculator
    reco::Photon::FiducialFlags fiducialFlags;
    reco::Photon::IsolationVariables isolVarR03, isolVarR04;
    if( thedet != DetId::Forward && thedet != DetId::Hcal) {
      thePhotonIsolationCalculator_->calculate( &newCandidate,evt,es,fiducialFlags,isolVarR04, isolVarR03);
    }
    newCandidate.setFiducialVolumeFlags( fiducialFlags );
    newCandidate.setIsolationVariables(isolVarR04, isolVarR03 );

        
    /// fill shower shape block
    reco::Photon::ShowerShape  showerShape;
    showerShape.e1x5= e1x5;
    showerShape.e2x5= e2x5;
    showerShape.e3x3= e3x3;
    showerShape.e5x5= e5x5;
    showerShape.maxEnergyXtal =  maxXtal;
    showerShape.sigmaEtaEta =    sigmaEtaEta;
    showerShape.sigmaIetaIeta =  sigmaIetaIeta;
    showerShape.hcalDepth1OverEcal = HoE1;
    showerShape.hcalDepth2OverEcal = HoE2;
    showerShape.hcalDepth1OverEcalBc = hcalDepth1OverEcalBc;
    showerShape.hcalDepth2OverEcalBc = hcalDepth2OverEcalBc;
    showerShape.hcalTowersBehindClusters =  TowersBehindClus;
    /// fill extra shower shapes
    const float spp = (!edm::isFinite(locCov[2]) ? 0. : sqrt(locCov[2]));
    const float sep = locCov[1];
    showerShape.sigmaIetaIphi = sep;
    showerShape.sigmaIphiIphi = spp;
    showerShape.e2nd          = ( hits != nullptr ? EcalClusterTools::e2nd(*(scRef->seed()),&(*hits)) : 0.f );
    showerShape.eTop          = ( hits != nullptr ? EcalClusterTools::eTop(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    showerShape.eLeft         = ( hits != nullptr ? EcalClusterTools::eLeft(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    showerShape.eRight        = ( hits != nullptr ? EcalClusterTools::eRight(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    showerShape.eBottom       = ( hits != nullptr ? EcalClusterTools::eBottom(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    showerShape.e1x3          = ( hits != nullptr ? EcalClusterTools::e1x3(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    showerShape.e2x2          = ( hits != nullptr ? EcalClusterTools::e2x2(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    showerShape.e2x5Max       = ( hits != nullptr ? EcalClusterTools::e2x5Max(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    showerShape.e2x5Left      = ( hits != nullptr ? EcalClusterTools::e2x5Left(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    showerShape.e2x5Right     = ( hits != nullptr ? EcalClusterTools::e2x5Right(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    showerShape.e2x5Top       = ( hits != nullptr ? EcalClusterTools::e2x5Top(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    showerShape.e2x5Bottom    = ( hits != nullptr ? EcalClusterTools::e2x5Bottom(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    // fill preshower shapes
    EcalClusterLazyTools toolsforES(evt, es, barrelEcalHits_, endcapEcalHits_, preshowerHits_);
    const float sigmaRR =  toolsforES.eseffsirir( *scRef );
    showerShape.effSigmaRR = sigmaRR;
    newCandidate.setShowerShapeVariables ( showerShape ); 

    reco::Photon::SaturationInfo saturationInfo;
    const reco::CaloCluster& seedCluster = *(scRef->seed()) ;
    DetId seedXtalId = seedCluster.seed();
    int nSaturatedXtals = 0;
    bool isSeedSaturated = false;
    if (hits != nullptr) {
      const auto hitsAndFractions = scRef->hitsAndFractions();
      for (auto&& hitFractionPair : hitsAndFractions) {    
	auto&& ecalRecHit = hits->find(hitFractionPair.first);
	if (ecalRecHit == hits->end()) continue;
	if (ecalRecHit->checkFlag(EcalRecHit::Flags::kSaturated)) {
	  nSaturatedXtals++;
	  if (seedXtalId == ecalRecHit->detid())
	    isSeedSaturated = true;
	}
      }
    }
    saturationInfo.nSaturatedXtals = nSaturatedXtals;
    saturationInfo.isSeedSaturated = isSeedSaturated;
    newCandidate.setSaturationInfo(saturationInfo);
    
    /// fill full5x5 shower shape block
    reco::Photon::ShowerShape  full5x5_showerShape;
    full5x5_showerShape.e1x5= full5x5_e1x5;
    full5x5_showerShape.e2x5= full5x5_e2x5;
    full5x5_showerShape.e3x3= full5x5_e3x3;
    full5x5_showerShape.e5x5= full5x5_e5x5;
    full5x5_showerShape.maxEnergyXtal =  full5x5_maxXtal;
    full5x5_showerShape.sigmaEtaEta =    full5x5_sigmaEtaEta;
    full5x5_showerShape.sigmaIetaIeta =  full5x5_sigmaIetaIeta;
    /// fill extra full5x5 shower shapes
    const float full5x5_spp = (!edm::isFinite(full5x5_locCov[2]) ? 0. : sqrt(full5x5_locCov[2]));
    const float full5x5_sep = full5x5_locCov[1];
    full5x5_showerShape.sigmaIetaIphi = full5x5_sep;
    full5x5_showerShape.sigmaIphiIphi = full5x5_spp;
    full5x5_showerShape.e2nd          = ( hits != nullptr ? noZS::EcalClusterTools::e2nd(*(scRef->seed()),&(*hits)) : 0.f );
    full5x5_showerShape.eTop          = ( hits != nullptr ? noZS::EcalClusterTools::eTop(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    full5x5_showerShape.eLeft         = ( hits != nullptr ? noZS::EcalClusterTools::eLeft(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    full5x5_showerShape.eRight        = ( hits != nullptr ? noZS::EcalClusterTools::eRight(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    full5x5_showerShape.eBottom       = ( hits != nullptr ? noZS::EcalClusterTools::eBottom(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    full5x5_showerShape.e1x3          = ( hits != nullptr ? noZS::EcalClusterTools::e1x3(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    full5x5_showerShape.e2x2          = ( hits != nullptr ? noZS::EcalClusterTools::e2x2(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    full5x5_showerShape.e2x5Max       = ( hits != nullptr ? noZS::EcalClusterTools::e2x5Max(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    full5x5_showerShape.e2x5Left      = ( hits != nullptr ? noZS::EcalClusterTools::e2x5Left(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    full5x5_showerShape.e2x5Right     = ( hits != nullptr ? noZS::EcalClusterTools::e2x5Right(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    full5x5_showerShape.e2x5Top       = ( hits != nullptr ? noZS::EcalClusterTools::e2x5Top(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
    full5x5_showerShape.e2x5Bottom    = ( hits != nullptr ? noZS::EcalClusterTools::e2x5Bottom(*(scRef->seed()), &(*hits), &(*topology)) : 0.f );
     // fill preshower shapes
    full5x5_showerShape.effSigmaRR = sigmaRR;
    newCandidate.full5x5_setShowerShapeVariables ( full5x5_showerShape );     
    
    

    /// get ecal photon specific corrected energy 
    /// plus values from regressions     and store them in the Photon
    // Photon candidate takes by default (set in photons_cfi.py) 
    // a 4-momentum derived from the ecal photon-specific corrections. 
    if( thedet != DetId::Forward && thedet != DetId::Hcal) {
      thePhotonEnergyCorrector_->calculate(evt, newCandidate, subdet, vertexCollection, es);
      if ( candidateP4type_ == "fromEcalEnergy") {
	newCandidate.setP4( newCandidate.p4(reco::Photon::ecal_photons) );
	newCandidate.setCandidateP4type(reco::Photon::ecal_photons);
      } else if ( candidateP4type_ == "fromRegression1") {
	newCandidate.setP4( newCandidate.p4(reco::Photon::regression1) );
	newCandidate.setCandidateP4type(reco::Photon::regression1);
      } else if ( candidateP4type_ == "fromRegression2") {
	newCandidate.setP4( newCandidate.p4(reco::Photon::regression2) );
	newCandidate.setCandidateP4type(reco::Photon::regression2);
      } else if ( candidateP4type_ == "fromRefinedSCRegression" ) {
	newCandidate.setP4( newCandidate.p4(reco::Photon::regression2) );
	newCandidate.setCandidateP4type(reco::Photon::regression2);
      }
    }

    //       std::cout << " final p4 " << newCandidate.p4() << " energy " << newCandidate.energy() <<  std::endl;


    // std::cout << " GEDPhotonProducer from candidate HoE with towers in a cone " << newCandidate.hadronicOverEm()  << "  " <<  newCandidate.hadronicDepth1OverEm()  << " " <<  newCandidate.hadronicDepth2OverEm()  << std::endl;
    //    std::cout << " GEDPhotonProducer from candidate  of HoE with towers behind the BCs " <<  newCandidate.hadTowOverEm()  << "  " << newCandidate.hadTowDepth1OverEm() << " " << newCandidate.hadTowDepth2OverEm() << std::endl;


  // fill MIP Vairables for Halo: Block for MIP are filled from PhotonMIPHaloTagger
   reco::Photon::MIPVariables mipVar ;
   if(subdet==EcalBarrel && runMIPTagger_ )
    {
  
     thePhotonMIPHaloTagger_-> MIPcalculate( &newCandidate,evt,es,mipVar);
    newCandidate.setMIPVariables(mipVar);
    }



    /// Pre-selection loose  isolation cuts
    bool isLooseEM=true; 
    if ( newCandidate.pt() < highEt_) { 
      if ( newCandidate.hadronicOverEm()                   >= preselCutValues[1] )                                            isLooseEM=false;
      if ( newCandidate.ecalRecHitSumEtConeDR04()          > preselCutValues[2]+ preselCutValues[3]*newCandidate.pt() )       isLooseEM=false;
      if ( newCandidate.hcalTowerSumEtConeDR04()           > preselCutValues[4]+ preselCutValues[5]*newCandidate.pt() )       isLooseEM=false;
      if ( newCandidate.nTrkSolidConeDR04()                > int(preselCutValues[6]) )                                        isLooseEM=false;
      if ( newCandidate.nTrkHollowConeDR04()               > int(preselCutValues[7]) )                                        isLooseEM=false;
      if ( newCandidate.trkSumPtSolidConeDR04()            > preselCutValues[8] )                                             isLooseEM=false;
      if ( newCandidate.trkSumPtHollowConeDR04()           > preselCutValues[9] )                                             isLooseEM=false;
      if ( newCandidate.sigmaIetaIeta()                    > preselCutValues[10] )                                            isLooseEM=false;
    } 
    

        
    if ( isLooseEM)  
      outputPhotonCollection.push_back(newCandidate);
      
        
  }
}




void GEDPhotonProducer::fillPhotonCollection(edm::Event& evt,
					     edm::EventSetup const & es,
					     const edm::Handle<reco::PhotonCollection> & photonHandle,
					     const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle,
					     const edm::Handle<reco::PFCandidateCollection> pfEGCandidateHandle,
					     edm::ValueMap<reco::PhotonRef> pfEGCandToPhotonMap,
					     edm::Handle< reco::VertexCollection >  & vertexHandle,
					     reco::PhotonCollection & outputPhotonCollection, int& iSC, const edm::Handle<edm::ValueMap<float>>& chargedHadrons_, const edm::Handle<edm::ValueMap<float>>& neutralHadrons_, const edm::Handle<edm::ValueMap<float>>& photons_) {

  
 
  std::vector<double> preselCutValues;


  for(unsigned int lSC=0; lSC < photonHandle->size(); lSC++) {
    reco::PhotonRef phoRef(reco::PhotonRef(photonHandle, lSC));
    reco::SuperClusterRef parentSCRef = phoRef->parentSuperCluster();
    reco::SuperClusterRef scRef=phoRef->superCluster();
    int thedet = scRef->seed()->hitsAndFractions()[0].first.det();
    int subdet = scRef->seed()->hitsAndFractions()[0].first.subdetId();
    if (subdet==EcalBarrel) { 
      preselCutValues = preselCutValuesBarrel_;
    } else if (subdet==EcalEndcap)  { 
      preselCutValues = preselCutValuesEndcap_;
    } else if ( thedet == DetId::Forward || thedet == DetId::Hcal) {
      preselCutValues = preselCutValuesEndcap_;
    } else {
      edm::LogWarning("")<<"GEDPhotonProducer: do not know if it is a barrel or endcap SuperCluster" << thedet << ' ' << subdet; 
    }


  
    // SC energy preselection
    if (parentSCRef.isNonnull() &&
	ptFast(parentSCRef->energy(),parentSCRef->position(),math::XYZPoint(0,0,0)) <= preselCutValues[0] ) continue;
    reco::Photon newCandidate(*phoRef);
    iSC++;    
  

  // Calculate the PF isolation and ID - for the time being there is no calculation. Only the setting
    reco::Photon::PflowIsolationVariables pfIso;
    reco::Photon::PflowIDVariables pfID;
  
    //get the pointer for the photon object
    edm::Ptr<reco::Photon> photonPtr(photonHandle, lSC);

    pfIso.chargedHadronIso = (*chargedHadrons_)[photonPtr] ;
    pfIso.neutralHadronIso = (*neutralHadrons_)[photonPtr];
    pfIso.photonIso        = (*photons_)[photonPtr];
    newCandidate.setPflowIsolationVariables(pfIso);
    newCandidate.setPflowIDVariables(pfID);


    // do the regression
    thePhotonEnergyCorrector_->calculate(evt, newCandidate, subdet, *vertexHandle, es);
    if ( candidateP4type_ == "fromEcalEnergy") {
      newCandidate.setP4( newCandidate.p4(reco::Photon::ecal_photons) );
      newCandidate.setCandidateP4type(reco::Photon::ecal_photons);
    } else if ( candidateP4type_ == "fromRegression1") {
      newCandidate.setP4( newCandidate.p4(reco::Photon::regression1) );
      newCandidate.setCandidateP4type(reco::Photon::regression1);
    } else if ( candidateP4type_ == "fromRegression2") {
      newCandidate.setP4( newCandidate.p4(reco::Photon::regression2) );
      newCandidate.setCandidateP4type(reco::Photon::regression2);
    } else if ( candidateP4type_ == "fromRefinedSCRegression" ) {
      newCandidate.setP4( newCandidate.p4(reco::Photon::regression2) );
      newCandidate.setCandidateP4type(reco::Photon::regression2);
    }

    //    std::cout << " GEDPhotonProducer  pf based isolation  chargedHadron " << newCandidate.chargedHadronIso() << " neutralHadron " <<  newCandidate.neutralHadronIso() << " Photon " <<  newCandidate.photonIso() << std::endl;
    //std::cout << " GEDPhotonProducer from candidate HoE with towers in a cone " << newCandidate.hadronicOverEm()  << "  " <<  newCandidate.hadronicDepth1OverEm()  << " " <<  newCandidate.hadronicDepth2OverEm()  << std::endl;
    //std::cout << " GEDPhotonProducer from candidate  of HoE with towers behind the BCs " <<  newCandidate.hadTowOverEm()  << "  " << newCandidate.hadTowDepth1OverEm() << " " << newCandidate.hadTowDepth2OverEm() << std::endl;
    //std::cout << " standard p4 before " << newCandidate.p4() << " energy " << newCandidate.energy() <<  std::endl;
    //std::cout << " type " <<newCandidate.getCandidateP4type() <<  " standard p4 after " << newCandidate.p4() << " energy " << newCandidate.energy() << std::endl;
    //std::cout << " final p4 " << newCandidate.p4() << " energy " << newCandidate.energy() <<  std::endl;

    outputPhotonCollection.push_back(newCandidate);        
    
  }

}
