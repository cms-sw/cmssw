#include "RecoEcal/EgammaClusterProducers/interface/EgammaSCCorrectionMaker.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h" 

#include <string>

EgammaSCCorrectionMaker::EgammaSCCorrectionMaker(const edm::ParameterSet& ps)
{
 

  // the input producers
  rHInputProducer_ = ps.getParameter<edm::InputTag>("recHitProducer");
  sCInputProducer_ = ps.getParameter<edm::InputTag>("rawSuperClusterProducer");
  std::string sCAlgo_str = ps.getParameter<std::string>("superClusterAlgo");

  // determine which BasicCluster algo we are correcting for
  //And obtain forrection parameters form cfg file
  edm::ParameterSet fCorrPset;
  if (sCAlgo_str=="Hybrid") {
    sCAlgo_= reco::CaloCluster::hybrid;
    fCorrPset = ps.getParameter<edm::ParameterSet>("hyb_fCorrPset"); 
  } else if (sCAlgo_str=="Island") {
    sCAlgo_= reco::CaloCluster::island;
    fCorrPset = ps.getParameter<edm::ParameterSet>("isl_fCorrPset");
  } else if (sCAlgo_str=="DynamicHybrid") {
    sCAlgo_ = reco::CaloCluster::dynamicHybrid;
    fCorrPset = ps.getParameter<edm::ParameterSet>("dyn_fCorrPset"); 
  } else if (sCAlgo_str=="Multi5x5") {
    sCAlgo_ = reco::CaloCluster::multi5x5;
    fCorrPset = ps.getParameter<edm::ParameterSet>("fix_fCorrPset");
  } else {
    edm::LogError("EgammaSCCorrectionMakerError") 
      << "Error! SuperClusterAlgo in config file must be Hybrid or Island: " 
      << sCAlgo_str << "  Using Hybrid by default";
    sCAlgo_=reco::CaloCluster::hybrid;
  }
  
  // set correction algo parameters
  applyEnergyCorrection_ = ps.getParameter<bool>("applyEnergyCorrection");
  applyCrackCorrection_  = ps.getParameter<bool>("applyCrackCorrection");
  applyLocalContCorrection_= ps.getParameter<bool>("applyLocalContCorrection");

  energyCorrectorName_ = ps.getParameter<std::string>("energyCorrectorName");
  crackCorrectorName_  = ps.getParameter<std::string>("crackCorrectorName");
  localContCorrectorName_= ps.getParameter<std::string>("localContCorrectorName");

  modeEB_ =  ps.getParameter<int>("modeEB");
  modeEE_ =  ps.getParameter<int>("modeEE");

  sigmaElectronicNoise_ =  ps.getParameter<double>("sigmaElectronicNoise");

  etThresh_ =  ps.getParameter<double>("etThresh");



  // set the producer parameters
  outputCollection_ = ps.getParameter<std::string>("corectedSuperClusterCollection");
  produces<reco::SuperClusterCollection>(outputCollection_);

  // instanciate the correction algo object
  energyCorrector_ = new EgammaSCEnergyCorrectionAlgo(sigmaElectronicNoise_, sCAlgo_, fCorrPset);
  
  // energy correction class
  if (applyEnergyCorrection_ )
    energyCorrectionFunction_ = EcalClusterFunctionFactory::get()->create(energyCorrectorName_.c_str(), ps);
    //energyCorrectionFunction_ = EcalClusterFunctionFactory::get()->create("EcalClusterEnergyCorrection", ps);
  else
    energyCorrectionFunction_=0;

  if (applyCrackCorrection_ )
    crackCorrectionFunction_ = EcalClusterFunctionFactory::get()->create(crackCorrectorName_, ps);
  else
    crackCorrectionFunction_=0;

  
  if (applyLocalContCorrection_ )
    localContCorrectionFunction_ = EcalClusterFunctionFactory::get()->create(localContCorrectorName_, ps);
  else
    localContCorrectionFunction_=0;

}

EgammaSCCorrectionMaker::~EgammaSCCorrectionMaker()
{
  if (energyCorrectionFunction_)    delete energyCorrectionFunction_;
  if (crackCorrectionFunction_)     delete crackCorrectionFunction_;
  if (localContCorrectionFunction_) delete localContCorrectionFunction_;

  delete energyCorrector_;
}

void
EgammaSCCorrectionMaker::produce(edm::Event& evt, const edm::EventSetup& es)
{
  using namespace edm;

  // initialize energy correction class
  if(applyEnergyCorrection_) 
    energyCorrectionFunction_->init(es);

  // initialize energy correction class
  if(applyCrackCorrection_) 
    crackCorrectionFunction_->init(es);
  

  // initialize containemnt correction class
  if(applyLocalContCorrection_) 
    localContCorrectionFunction_->init(es);

  // get the collection geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry *geometry_p;

  std::string rHInputCollection = rHInputProducer_.instance();
  if(rHInputCollection == "EcalRecHitsEB") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  } else if(rHInputCollection == "EcalRecHitsEE") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  } else if(rHInputCollection == "EcalRecHitsPS") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  } else {
          std::string str = "\n\nSCCorrectionMaker encountered invalied ecalhitcollection type: " + rHInputCollection + ".\n\n";
          throw(std::runtime_error( str.c_str() ));
  }
  
  // Get raw SuperClusters from the event    
  Handle<reco::SuperClusterCollection> pRawSuperClusters;
  try { 
    evt.getByLabel(sCInputProducer_, pRawSuperClusters);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EgammaSCCorrectionMakerError") 
      << "Error! can't get the rawSuperClusters " 
      << sCInputProducer_.label() ;
  }    
  
  // Get the RecHits from the event
  Handle<EcalRecHitCollection> pRecHits;
  try { 
    evt.getByLabel(rHInputProducer_, pRecHits);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EgammaSCCorrectionMakerError") 
      << "Error! can't get the RecHits " 
      << rHInputProducer_.label();
  }    
  
  // Create a pointer to the RecHits and raw SuperClusters
  const EcalRecHitCollection *hitCollection = pRecHits.product();
  const reco::SuperClusterCollection *rawClusters = pRawSuperClusters.product();
   
  // Define a collection of corrected SuperClusters to put back into the event
  std::auto_ptr<reco::SuperClusterCollection> corrClusters(new reco::SuperClusterCollection);
  
  //  Loop over raw clusters and make corrected ones
  reco::SuperClusterCollection::const_iterator aClus;
  for(aClus = rawClusters->begin(); aClus != rawClusters->end(); aClus++)
    {
      reco::SuperCluster enecorrClus,crackcorrClus,localContCorrClus;



      if(applyCrackCorrection_)
	crackcorrClus=energyCorrector_->applyCrackCorrection(*aClus,crackCorrectionFunction_);
      else 
	crackcorrClus=*aClus;

      if (applyLocalContCorrection_)
	localContCorrClus = 
	  energyCorrector_->applyLocalContCorrection(crackcorrClus,localContCorrectionFunction_);
      else
	localContCorrClus = crackcorrClus;
      

      if(applyEnergyCorrection_) 
        enecorrClus = energyCorrector_->applyCorrection(localContCorrClus, *hitCollection, sCAlgo_, geometry_p, energyCorrectionFunction_, energyCorrectorName_, modeEB_, modeEE_);
      else
	enecorrClus=localContCorrClus;


      if(enecorrClus.energy()*sin(enecorrClus.position().theta())>etThresh_) {
	
	corrClusters->push_back(enecorrClus);
      }
    }

  // Put collection of corrected SuperClusters into the event
  evt.put(corrClusters, outputCollection_);   
  
}

