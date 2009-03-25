#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

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

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonProducer.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"


PhotonProducer::PhotonProducer(const edm::ParameterSet& config) : 
  conf_(config)
{

  // use onfiguration file to setup input/output collection names

  photonCoreProducer_   = conf_.getParameter<std::string>("photonCoreProducer");
  barrelEcalHits_   = conf_.getParameter<edm::InputTag>("barrelEcalHits");
  endcapEcalHits_   = conf_.getParameter<edm::InputTag>("endcapEcalHits");
  vertexProducer_   = conf_.getParameter<std::string>("primaryVertexProducer");
  PhotonCoreCollection_ = conf_.getParameter<std::string>("photonCollection");
  hcalTowers_ = conf_.getParameter<edm::InputTag>("hcalTowers");
  hOverEConeSize_   = conf_.getParameter<double>("hOverEConeSize");
  highEt_        = conf_.getParameter<double>("highEt");
  // R9 value to decide converted/unconverted
  minR9Barrel_        = conf_.getParameter<double>("minR9Barrel");
  minR9Endcap_        = conf_.getParameter<double>("minR9Endcap");
  usePrimaryVertex_ = conf_.getParameter<bool>("usePrimaryVertex");
 
 
  // Parameters for the position calculation:
  std::map<std::string,double> providedParameters;
  providedParameters.insert(std::make_pair("LogWeighted",conf_.getParameter<bool>("posCalc_logweight")));
  providedParameters.insert(std::make_pair("T0_barl",conf_.getParameter<double>("posCalc_t0_barl")));
  providedParameters.insert(std::make_pair("T0_endc",conf_.getParameter<double>("posCalc_t0_endc")));
  providedParameters.insert(std::make_pair("T0_endcPresh",conf_.getParameter<double>("posCalc_t0_endcPresh")));
  providedParameters.insert(std::make_pair("W0",conf_.getParameter<double>("posCalc_w0")));
  providedParameters.insert(std::make_pair("X0",conf_.getParameter<double>("posCalc_x0")));
  posCalculator_ = PositionCalc(providedParameters);
  // cut values for pre-selection
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("minSCEtBarrel")); 
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("maxHoverEBarrel")); 
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("ecalRecHitSumEtBarrel")); 
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("hcalTowerSumEtBarrel"));
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("nTrackSolidConeBarrel"));
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("nTrackHollowConeBarrel"));     
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("trackPtSumSolidConeBarrel"));     
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("trackPtSumHollowConeBarrel"));     
  preselCutValuesBarrel_.push_back(conf_.getParameter<double>("sigmaIetaIetaCutBarrel"));     
  //  
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("minSCEtEndcap")); 
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("maxHoverEEndcap")); 
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("ecalRecHitSumEtEndcap")); 
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("hcalTowerSumEtEndcap"));
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("nTrackSolidConeEndcap"));
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("nTrackHollowConeEndcap"));     
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("trackPtSumSolidConeEndcap"));     
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("trackPtSumHollowConeEndcap"));     
  preselCutValuesEndcap_.push_back(conf_.getParameter<double>("sigmaIetaIetaCutEndcap"));     
  //
  thePhotonIsolationCalculator_ = new PhotonIsolationCalculator();
  edm::ParameterSet isolationSumsCalculatorSet = conf_.getParameter<edm::ParameterSet>("isolationSumsCalculatorSet"); 
  thePhotonIsolationCalculator_->setup(isolationSumsCalculatorSet);


  // Register the product
  produces< reco::PhotonCollection >(PhotonCollection_);

}

PhotonProducer::~PhotonProducer() {}



void  PhotonProducer::beginRun (edm::Run& r, edm::EventSetup const & theEventSetup) {

}

void  PhotonProducer::endRun (edm::Run& r, edm::EventSetup const & theEventSetup) {

  delete thePhotonIsolationCalculator_;

}




void PhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  using namespace edm;
  //  nEvt_++;

  reco::PhotonCollection outputPhotonCollection;
  std::auto_ptr< reco::PhotonCollection > outputPhotonCollection_p(new reco::PhotonCollection);


  // Get the PhotonCore collection
  bool validPhotonCoreHandle=true;
  Handle<reco::PhotonCoreCollection> photonCoreHandle;
  theEvent.getByLabel(photonCoreProducer_,photonCoreHandle);
  if (!photonCoreHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<photonCoreProducer_.label();
    validPhotonCoreHandle=false;
  }

 // Get EcalRecHits
  bool validEcalRecHits=true;
  Handle<EcalRecHitCollection> barrelHitHandle;
  EcalRecHitCollection barrelRecHits;
  theEvent.getByLabel(barrelEcalHits_, barrelHitHandle);
  if (!barrelHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<barrelEcalHits_.label();
    validEcalRecHits=false; 
  }
  if (  validEcalRecHits)  barrelRecHits = *(barrelHitHandle.product());

  
  Handle<EcalRecHitCollection> endcapHitHandle;
  theEvent.getByLabel(endcapEcalHits_, endcapHitHandle);
  EcalRecHitCollection endcapRecHits;
  if (!endcapHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<endcapEcalHits_.label();
    validEcalRecHits=false; 
  }
  if( validEcalRecHits) endcapRecHits = *(endcapHitHandle.product());


// get Hcal towers collection 
  Handle<CaloTowerCollection> hcalTowersHandle;
  theEvent.getByLabel(hcalTowers_, hcalTowersHandle);


  // get the geometry from the event setup:
  theEventSetup.get<CaloGeometryRecord>().get(theCaloGeom_);
  

  edm::ESHandle<CaloTopology> pTopology;
  theEventSetup.get<CaloTopologyRecord>().get(theCaloTopo_);
  const CaloTopology *topology = theCaloTopo_.product();

  // Get the primary event vertex
  Handle<reco::VertexCollection> vertexHandle;
  reco::VertexCollection vertexCollection;
  bool validVertex=true;
  if ( usePrimaryVertex_ ) {
    theEvent.getByLabel(vertexProducer_, vertexHandle);
    if (!vertexHandle.isValid()) {
      edm::LogError("PhotonProducer") << "Error! Can't get the product primary Vertex Collection "<< "\n";
      validVertex=false;
    }
    if (validVertex) vertexCollection = *(vertexHandle.product());
  }
  math::XYZPoint vtx(0.,0.,0.);
  if (vertexCollection.size()>0) vtx = vertexCollection.begin()->position();


  int iSC=0; // index in photon collection
  // Loop over barrel and endcap SC collections and fill the  photon collection
  if ( validPhotonCoreHandle) 
    fillPhotonCollection(theEvent,
			 theEventSetup,
			 photonCoreHandle,
			 topology,
			 &barrelRecHits,
			 &endcapRecHits,
			 hcalTowersHandle,
			 vtx,
			 outputPhotonCollection,
			 iSC);
 

  // put the product in the event
  edm::LogInfo("PhotonProducer") << " Put in the event " << iSC << " Photon Candidates \n";
  outputPhotonCollection_p->assign(outputPhotonCollection.begin(),outputPhotonCollection.end());
  theEvent.put( outputPhotonCollection_p, PhotonCollection_);

}

void PhotonProducer::fillPhotonCollection(edm::Event& evt,
					  edm::EventSetup const & es,
					  const edm::Handle<reco::PhotonCoreCollection> & photonCoreHandle,
					  const CaloTopology* topology,
					  const EcalRecHitCollection* ecalBarrelHits,
					  const EcalRecHitCollection* ecalEndcapHits,
					  const edm::Handle<CaloTowerCollection> & hcalTowersHandle, 
					  math::XYZPoint & vtx,
					  reco::PhotonCollection & outputPhotonCollection, int& iSC) {
  
  const CaloGeometry* geometry = theCaloGeom_.product();
  const CaloSubdetectorGeometry* subDetGeometry =0 ;
  const CaloSubdetectorGeometry* geometryES = theCaloGeom_->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  const EcalRecHitCollection* hits = 0 ;
  std::vector<double> preselCutValues;
  float minR9=0;



  for(unsigned int lSC=0; lSC < photonCoreHandle->size(); lSC++) {

    reco::PhotonCoreRef coreRef(reco::PhotonCoreRef(photonCoreHandle, lSC));
    reco::SuperClusterRef scRef=coreRef->superCluster();
    iSC++;

    int subdet = scRef->seed()->hitsAndFractions()[0].first.subdetId();
    subDetGeometry =  theCaloGeom_->getSubdetectorGeometry(DetId::Ecal, subdet);

    if (subdet==EcalBarrel) 
      { 
	preselCutValues = preselCutValuesBarrel_;
        minR9=minR9Barrel_;
        hits=  ecalBarrelHits;
	std::cout << " Photon Producer : Barrel " << std::endl;
      }
    else if  (subdet==EcalEndcap) 
      { 
	preselCutValues = preselCutValuesEndcap_;
        minR9=minR9Endcap_;
	hits=  ecalEndcapHits;
	std::cout << " Photon Producer : Endcap " << std::endl;
      }
    else
      { edm::LogWarning("")<<"PhotonProducer: do not know if it is a barrel or endcap SuperCluster" ; }
    
        
    // SC energy preselection
    if (scRef->energy()/cosh(scRef->eta()) <= preselCutValues[0] ) continue;
    // calculate HoE
    const CaloTowerCollection* hcalTowersColl = hcalTowersHandle.product();
    EgammaTowerIsolation towerIso1(hOverEConeSize_,0.,0.,1,hcalTowersColl) ;  
    EgammaTowerIsolation towerIso2(hOverEConeSize_,0.,0.,2,hcalTowersColl) ;  
    double HoE1=towerIso1.getTowerESum(&(*scRef))/scRef->energy();
    double HoE2=towerIso2.getTowerESum(&(*scRef))/scRef->energy(); 
    //     std::cout << " PhotonProducer " << HoE1  << "  HoE2 " << HoE2 << std::endl;
    //std::cout << " PhotonProducer calcualtion of HoE1 " << HoE1  << "  HoE2 " << HoE2 << std::endl;

    
    // recalculate position of seed BasicCluster taking shower depth for unconverted photon
    math::XYZPoint unconvPos = posCalculator_.Calculate_Location(scRef->seed()->hitsAndFractions(),hits,subDetGeometry,geometryES);

    static std::pair<DetId, float> maxXtal = EcalClusterTools::getMaximum (*(scRef->seed()), &(*hits) );
    float e1x5    =   EcalClusterTools::e1x5(  *(scRef->seed()), &(*hits), &(*topology)); 
    float e2x5    =   EcalClusterTools::e2x5Max(  *(scRef->seed()), &(*hits), &(*topology)); 
    float e3x3    =   EcalClusterTools::e3x3(  *(scRef->seed()), &(*hits), &(*topology)); 
    float e5x5    =   EcalClusterTools::e5x5( *(scRef->seed()), &(*hits), &(*topology)); 
    std::vector<float> cov =  EcalClusterTools::covariances( *(scRef->seed()), &(*hits), &(*topology), geometry); 
    float sigmaEtaEta = sqrt(cov[0]);
    std::vector<float> locCov =  EcalClusterTools::localCovariances( *(scRef->seed()), &(*hits), &(*topology)); 
    float sigmaIetaIeta = sqrt(locCov[0]);


    float r9 =e3x3/(scRef->rawEnergy());
    // compute position of ECAL shower
    math::XYZPoint caloPosition;
    double photonEnergy=0;
    if (r9>minR9) {
      caloPosition = unconvPos;
      photonEnergy=e5x5 + scRef->preshowerEnergy() ;
    } else {
      caloPosition = scRef->position();
      photonEnergy=scRef->energy();
    }
    
    
    // compute momentum vector of photon from primary vertex and cluster position
    math::XYZVector direction = caloPosition - vtx;
    math::XYZVector momentum = direction.unit() * photonEnergy ;

    // Create candidate
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), photonEnergy );
    reco::Photon newCandidate(p4, caloPosition, coreRef, vtx);

    // Calculate fiducial flags and isolation variable. Blocked are filled from the isolationCalculator
    reco::Photon::FiducialFlags fiducialFlags;
    reco::Photon::IsolationVariables isolVarR03, isolVarR04;
    thePhotonIsolationCalculator_-> calculate ( &newCandidate,evt,es,fiducialFlags,isolVarR04, isolVarR03);
    newCandidate.setFiducialVolumeFlags( fiducialFlags );
    newCandidate.setIsolationVariables(isolVarR04, isolVarR03 );
  
    /// fill shower shape block
    reco::Photon::ShowerShape  showerShape;
    showerShape.e1x5= e1x5;
    showerShape.e2x5= e2x5;
    showerShape.e3x3= e3x3;
    showerShape.e5x5= e5x5;
    showerShape.maxEnergyXtal =  maxXtal.second;
    showerShape.sigmaEtaEta =    sigmaEtaEta;
    showerShape.sigmaIetaIeta =  sigmaIetaIeta;
    showerShape.hcalDepth1OverEcal = HoE1;
    showerShape.hcalDepth2OverEcal = HoE2;
    newCandidate.setShowerShapeVariables ( showerShape ); 
    
    //std::cout << " PhotonProducer checking shower shapes e1x5  " << newCandidate.e1x5() << " r9 " << newCandidate.r9() << std::endl;
    //std::cout << " PhotonProducer checking fiducial flags is EB " << newCandidate.isEB() << " isEE() " << newCandidate.isEE() << std::endl;
    //std::cout << " PhotonProducer checking isolation hadronicOverEm " << newCandidate.hadronicOverEm()  << std::endl;
    //std::cout <<   " .ecalRecHitSumConeDR04 " << newCandidate.ecalRecHitSumEtConeDR04() << std::endl;
    //std::cout <<   " .hcalTowerSumConeDR04 " << newCandidate.hcalTowerSumEtConeDR04() << std::endl;
    //std::cout <<   " .nTrkSolidConeDR04() " << newCandidate.nTrkSolidConeDR04() << std::endl;
    //std::cout <<   " .nTrkHollowConeDR04() " << newCandidate.nTrkHollowConeDR04() << std::endl;
    //std::cout <<   " .sigmaIetaIeta() " << newCandidate.sigmaIetaIeta() << std::endl;

    /// Pre-selection loose  isolation cuts
    bool isLooseEM=true;
    
    if ( newCandidate.pt() < highEt_) { 
      if ( newCandidate.hadronicOverEm()                 >= preselCutValues[1] )      isLooseEM=false;
      if ( newCandidate.ecalRecHitSumEtConeDR04()        > preselCutValues[2] )       isLooseEM=false;
      if ( newCandidate.hcalTowerSumEtConeDR04()         > preselCutValues[3] )       isLooseEM=false;
      if ( newCandidate.nTrkSolidConeDR04()              > int(preselCutValues[4]) )  isLooseEM=false;
      if ( newCandidate.nTrkHollowConeDR04()             > int(preselCutValues[5]) )  isLooseEM=false;
      if ( newCandidate.trkSumPtSolidConeDR04()          > preselCutValues[6] )       isLooseEM=false;
      if ( newCandidate.trkSumPtHollowConeDR04()         > preselCutValues[7] )       isLooseEM=false;
      if ( newCandidate.sigmaIetaIeta()                  > preselCutValues[8] )       isLooseEM=false;
    } 
    
    
    if ( isLooseEM)  
      outputPhotonCollection.push_back(newCandidate);
      
        
  }
}

