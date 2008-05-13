#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"


#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"

#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonProducer.h"



PhotonProducer::PhotonProducer(const edm::ParameterSet& config) : 
  conf_(config), 
  theLikelihoodCalc_(0)

{

  // use onfiguration file to setup input/output collection names
  scHybridBarrelProducer_       = conf_.getParameter<edm::InputTag>("scHybridBarrelProducer");
  scIslandEndcapProducer_       = conf_.getParameter<edm::InputTag>("scIslandEndcapProducer");
  //  scHybridBarrelCollection_     = conf_.getParameter<std::string>("scHybridBarrelCollection");
  //  scIslandEndcapCollection_     = conf_.getParameter<std::string>("scIslandEndcapCollection");
  barrelEcalHits_   = conf_.getParameter<edm::InputTag>("barrelEcalHits");
  endcapEcalHits_   = conf_.getParameter<edm::InputTag>("endcapEcalHits");

  conversionProducer_ = conf_.getParameter<std::string>("conversionProducer");
  conversionCollection_ = conf_.getParameter<std::string>("conversionCollection");
  vertexProducer_   = conf_.getParameter<std::string>("primaryVertexProducer");
  PhotonCollection_ = conf_.getParameter<std::string>("photonCollection");
  pixelSeedProducer_   = conf_.getParameter<std::string>("pixelSeedProducer");

  hbheLabel_        = conf_.getParameter<std::string>("hbheModule");
  hbheInstanceName_ = conf_.getParameter<std::string>("hbheInstance");
  hOverEConeSize_   = conf_.getParameter<double>("hOverEConeSize");
  maxHOverE_        = conf_.getParameter<double>("maxHOverE");
  minSCEt_        = conf_.getParameter<double>("minSCEt");
  minR9_        = conf_.getParameter<double>("minR9");
  likelihoodWeights_= conf_.getParameter<std::string>("MVA_weights_location");


  usePrimaryVertex_ = conf_.getParameter<bool>("usePrimaryVertex");
  risolveAmbiguity_ = conf_.getParameter<bool>("risolveConversionAmbiguity");

 
  // Parameters for the position calculation:
  std::map<std::string,double> providedParameters;
  providedParameters.insert(std::make_pair("LogWeighted",conf_.getParameter<bool>("posCalc_logweight")));
  providedParameters.insert(std::make_pair("T0_barl",conf_.getParameter<double>("posCalc_t0_barl")));
  providedParameters.insert(std::make_pair("T0_endc",conf_.getParameter<double>("posCalc_t0_endc")));
  providedParameters.insert(std::make_pair("T0_endcPresh",conf_.getParameter<double>("posCalc_t0_endcPresh")));
  providedParameters.insert(std::make_pair("W0",conf_.getParameter<double>("posCalc_w0")));
  providedParameters.insert(std::make_pair("X0",conf_.getParameter<double>("posCalc_x0")));
  posCalculator_ = PositionCalc(providedParameters);

  // Register the product
  produces< reco::PhotonCollection >(PhotonCollection_);

}

PhotonProducer::~PhotonProducer() {
  delete theLikelihoodCalc_;
}


void  PhotonProducer::beginJob (edm::EventSetup const & theEventSetup) {
  theLikelihoodCalc_ = new ConversionLikelihoodCalculator();
  edm::FileInPath path_mvaWeightFile(likelihoodWeights_.c_str() );
  theLikelihoodCalc_->setWeightsFile(path_mvaWeightFile.fullPath().c_str());


}


void PhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  using namespace edm;

  reco::PhotonCollection outputPhotonCollection;
  std::auto_ptr< reco::PhotonCollection > outputPhotonCollection_p(new reco::PhotonCollection);

  // Get the  Barrel Super Cluster collection
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scBarrelHandle);
  if (!scBarrelHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<scHybridBarrelProducer_.label();
    return;
  }
  reco::SuperClusterCollection scBarrelCollection = *(scBarrelHandle.product());
  edm::LogInfo("PhotonProducer") << " Accessing Barrel SC collection with size : " << scBarrelCollection.size()  << "\n";

 // Get the  Endcap Super Cluster collection
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scEndcapHandle);
  if (!scEndcapHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<scIslandEndcapProducer_.label();
    return;
  }
  reco::SuperClusterCollection scEndcapCollection = *(scEndcapHandle.product());
  edm::LogInfo("PhotonProducer") << " Accessing Endcap SC collection with size : " << scEndcapCollection.size()  << "\n";
  

  // Get EcalRecHits
  Handle<EcalRecHitCollection> barrelHitHandle;
  theEvent.getByLabel(barrelEcalHits_, barrelHitHandle);
  if (!barrelHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<barrelEcalHits_.label();
    return;
  }
  const EcalRecHitCollection *barrelRecHits = barrelHitHandle.product();


  Handle<EcalRecHitCollection> endcapHitHandle;
  theEvent.getByLabel(endcapEcalHits_, endcapHitHandle);
  if (!endcapHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<endcapEcalHits_.label();
    return;
  }
  const EcalRecHitCollection *endcapRecHits = endcapHitHandle.product();

  // get the geometry from the event setup:
  theEventSetup.get<IdealGeometryRecord>().get(theCaloGeom_);
  const CaloSubdetectorGeometry *barrelGeometry = theCaloGeom_->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  const CaloSubdetectorGeometry *endcapGeometry = theCaloGeom_->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  const CaloSubdetectorGeometry *preshowerGeometry = theCaloGeom_->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

  edm::ESHandle<CaloTopology> pTopology;
  theEventSetup.get<CaloTopologyRecord>().get(theCaloTopo_);
  const CaloTopology *topology = theCaloTopo_.product();


  ///// Get the conversion collection
  validConversions_=true;
  edm::Handle<reco::ConversionCollection> conversionHandle; 
  theEvent.getByLabel(conversionProducer_, conversionCollection_ , conversionHandle);
  if (!conversionHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product  "<<conversionCollection_.c_str() << " but keep running. Corrected Photons will be made with null reference to conversions " << "\n";
    //return;
    validConversions_=false;
  }
 


  // Get HoverE
  Handle<HBHERecHitCollection> hbhe;
  std::auto_ptr<HBHERecHitMetaCollection> mhbhe;
  theEvent.getByLabel(hbheLabel_,hbheInstanceName_,hbhe);  
  if (!hbhe.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<hbheInstanceName_.c_str();
    return; 
  }

  if (hOverEConeSize_ > 0.) {
    mhbhe=  std::auto_ptr<HBHERecHitMetaCollection>(new HBHERecHitMetaCollection(*hbhe));
  }

  
  theHoverEcalc_=HoECalculator(theCaloGeom_);

  // Get ElectronPixelSeeds
  Handle<reco::ElectronPixelSeedCollection> pixelSeedHandle;
  theEvent.getByLabel(pixelSeedProducer_, pixelSeedHandle);
  if (!pixelSeedHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product ElectronPixelSeedHandle "<< "\n";
    return;
  }
  const reco::ElectronPixelSeedCollection& pixelSeeds = *pixelSeedHandle;

  // Get the primary event vertex
  Handle<reco::VertexCollection> vertexHandle;
  reco::VertexCollection vertexCollection;
  if ( usePrimaryVertex_ ) {
    theEvent.getByLabel(vertexProducer_, vertexHandle);
    if (!vertexHandle.isValid()) {
      edm::LogError("PhotonProducer") << "Error! Can't get the product primary Vertex Collection "<< "\n";
      return;
    }
    vertexCollection = *(vertexHandle.product());
  }
  math::XYZPoint vtx(0.,0.,0.);
  if (vertexCollection.size()>0) vtx = vertexCollection.begin()->position();

  edm::LogInfo("PhotonProducer") << "Constructing Photon 4-vectors assuming primary vertex position: " << vtx << std::endl;

  int iSC=0; // index in photon collection
  // Loop over barrel and endcap SC collections and fill the  photon collection
  fillPhotonCollection(scBarrelHandle,barrelGeometry,preshowerGeometry,topology,barrelRecHits,mhbhe.get(),conversionHandle,pixelSeeds,vtx,outputPhotonCollection,iSC);
  fillPhotonCollection(scEndcapHandle,endcapGeometry,preshowerGeometry,topology,endcapRecHits,mhbhe.get(),conversionHandle,pixelSeeds,vtx,outputPhotonCollection,iSC);

  // put the product in the event
  edm::LogInfo("PhotonProducer") << " Put in the event " << iSC << " Photon Candidates \n";
  outputPhotonCollection_p->assign(outputPhotonCollection.begin(),outputPhotonCollection.end());
  theEvent.put( outputPhotonCollection_p, PhotonCollection_);

}

void PhotonProducer::fillPhotonCollection(
		   const edm::Handle<reco::SuperClusterCollection> & scHandle,
		   const CaloSubdetectorGeometry *geometry,
		   const CaloSubdetectorGeometry *geometryES,
		   const CaloTopology *topology,
		   const EcalRecHitCollection *hits,
		   HBHERecHitMetaCollection *mhbhe,
                   const edm::Handle<reco::ConversionCollection> & conversionHandle,
		   const reco::ElectronPixelSeedCollection& pixelSeeds,
		   math::XYZPoint & vtx,
		   reco::PhotonCollection & outputPhotonCollection, int& iSC) {

  reco::SuperClusterCollection scCollection = *(scHandle.product());
  reco::SuperClusterCollection::iterator aClus;
  reco::ElectronPixelSeedCollection::const_iterator pixelSeedItr;

  reco::ConversionCollection conversionCollection;
  if (validConversions_) conversionCollection = *(conversionHandle.product());


  int lSC=0; // reset local supercluster index
  for(aClus = scCollection.begin(); aClus != scCollection.end(); aClus++) {

    // get SuperClusterRef
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scHandle, lSC));
    iSC++;
    lSC++;

    const reco::SuperCluster* pClus=&(*aClus);
    
    // preselection
    if (aClus->energy()/cosh(aClus->eta()) <= minSCEt_) continue;
    // calculate HoE
    double HoE=theHoverEcalc_(pClus,mhbhe);
    if (HoE>=maxHOverE_)  continue;
    
    
    
    // recalculate position of seed BasicCluster taking shower depth for unconverted photon
    math::XYZPoint unconvPos = posCalculator_.Calculate_Location(aClus->seed()->getHitsByDetId(),hits,geometry,geometryES);
    
    // compute position of ECAL shower
    float e3x3=   EcalClusterTools::e3x3(  *(aClus->seed()), &(*hits), &(*topology)); 
    float r9 =e3x3/(aClus->rawEnergy()+aClus->preshowerEnergy());
    float e5x5= EcalClusterTools::e5x5( *(aClus->seed()), &(*hits), &(*topology)); 

    math::XYZPoint caloPosition;
    double photonEnergy=0;
    if (r9>minR9_) {
      caloPosition = unconvPos;
      photonEnergy=e5x5;
    } else {
      caloPosition = aClus->position();
      photonEnergy=aClus->energy();
    }
    
    // does the SuperCluster have a matched pixel seed?
    bool hasSeed = false;
    for(pixelSeedItr = pixelSeeds.begin(); pixelSeedItr != pixelSeeds.end(); pixelSeedItr++) {
      if (fabs(pixelSeedItr->superCluster()->eta() - aClus->eta()) < 0.0001 &&
	  fabs(pixelSeedItr->superCluster()->phi() - aClus->phi()) < 0.0001) {
	hasSeed=true;
	break;
      }
    }
    
    // compute momentum vector of photon from primary vertex and cluster position
    math::XYZVector direction = caloPosition - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();

    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), photonEnergy );

    
    reco::Photon newCandidate(p4, caloPosition, scRef, HoE, hasSeed, vtx);

    if ( validConversions_) {

      
      if ( risolveAmbiguity_ ) { 
	
        reco::ConversionRef bestRef=solveAmbiguity( conversionHandle , scRef);	

	newCandidate.addConversion(bestRef);     

	
      } else {
	
	
	int icp=0;
	
	for( reco::ConversionCollection::const_iterator  itCP = conversionCollection.begin(); itCP != conversionCollection.end(); itCP++) {
	  
	  reco::ConversionRef cpRef(reco::ConversionRef(conversionHandle,icp));
	  icp++;      
	  
          if ( scRef != (*itCP).superCluster() ) continue; 
	  if ( !(*itCP).isConverted() ) continue;  
	  
	  
	  newCandidate.addConversion(cpRef);     
	}	  
	
      } // solve or not the ambiguity	     



      
    }

    outputPhotonCollection.push_back(newCandidate);
    
    
  }
  
}


reco::ConversionRef  PhotonProducer::solveAmbiguity(const edm::Handle<reco::ConversionCollection> & conversionHandle, reco::SuperClusterRef& scRef) {

  std::multimap<reco::ConversionRef, double >   convMap;
  int icp=0;
  reco::ConversionCollection conversionCollection  = *(conversionHandle.product());
  for( reco::ConversionCollection::const_iterator  itCP = conversionCollection.begin(); itCP != conversionCollection.end(); itCP++) {
    reco::ConversionRef cpRef(reco::ConversionRef(conversionHandle,icp));
    icp++;      
    
    if ( scRef != (*itCP).superCluster() ) continue; 
    if ( !(*itCP).isConverted() ) continue;  
    //  std::cout << " PhotonProducer conversion SC energy " << (*itCP).superCluster()->energy() << std::endl;
    
    double like = theLikelihoodCalc_->calculateLikelihood(cpRef);
    // another annoying cout that should not be in production. ShR 13 May 08
    //std::cout << " Like " << like << std::endl;
    convMap.insert ( std::make_pair(cpRef,like) ) ;
  }		     
  
  
  
  std::multimap<reco::ConversionRef, double >::iterator  iMap; 
  double max_lh = -1.;
  reco::ConversionRef bestRef;
  //std::cout << " Pick up the best conv " << std::endl;
  for (iMap=convMap.begin();  iMap!=convMap.end(); iMap++) {
    double like = iMap->second;
    if (like > max_lh) { 
      max_lh = like;
      bestRef=iMap->first;
    }
  }            
  
  //std::cout << " Best conv like " << max_lh << std::endl;    
  
  float ep=0;
  if ( max_lh <0 ) {
    //  std::cout << " Candidates with only one track " << std::endl;
    /// only one track reconstructed. Pick the one with best E/P
    float epMin=999; 
    
    for (iMap=convMap.begin();  iMap!=convMap.end(); iMap++) {
      reco::ConversionRef convRef=iMap->first;
      std::vector<reco::TrackRef> tracks = convRef->tracks();	
	    float px=tracks[0]->innerMomentum().x();
	    float py=tracks[0]->innerMomentum().y();
	    float pz=tracks[0]->innerMomentum().z();
	    float p=sqrt(px*px+py*py+pz*pz);
	    ep=fabs(1.-convRef->superCluster()->energy()/p);
	    //    std::cout << " 1-E/P = " << ep << std::endl;
	    if ( ep<epMin) {
	      epMin=ep;
	      bestRef=iMap->first;
	    }
    }
    //std::cout << " Best conv 1-E/P " << ep << std::endl;    
            
  }
  

  return bestRef;
  
  
} 

double PhotonProducer::hOverE(const reco::SuperClusterRef & scRef,
			      HBHERecHitMetaCollection *mhbhe){

  ////// this is obsolete. Taking the calculator in EgammaTools instead
  double HoE=0;
  if (mhbhe) {
    CaloConeSelector sel(hOverEConeSize_, theCaloGeom_.product(), DetId::Hcal);
    GlobalPoint pclu((*scRef).x(),(*scRef).y(),(*scRef).z());
    double hcalEnergy = 0.;
    std::auto_ptr<CaloRecHitMetaCollectionV> chosen=sel.select(pclu,*mhbhe);
    for (CaloRecHitMetaCollectionV::const_iterator i=chosen->begin(); i!=chosen->end(); i++) {
      hcalEnergy += i->energy();
    }
    HoE= hcalEnergy/(*scRef).energy();
    LogDebug("") << "H/E : " << HoE;
  }
  return HoE;
}
