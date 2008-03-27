#include "RecoEcal/EgammaClusterProducers/interface/EgammaSCCorrectionMaker.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"

#include <string>

EgammaSCCorrectionMaker::EgammaSCCorrectionMaker(const edm::ParameterSet& ps)
{
 
  // The verbosity level
  std::string debugString = ps.getParameter<std::string>("VerbosityLevel");
  if      (debugString == "DEBUG")   verbosity_ = EgammaSCEnergyCorrectionAlgo::pDEBUG;
  else if (debugString == "INFO")    verbosity_ = EgammaSCEnergyCorrectionAlgo::pINFO;
  else                               verbosity_ = EgammaSCEnergyCorrectionAlgo::pERROR;

  // the input producers
  rHInputProducer_ = ps.getParameter<edm::InputTag>("recHitProducer");
  sCInputProducer_ = ps.getParameter<edm::InputTag>("rawSuperClusterProducer");
  std::string sCAlgo_str = ps.getParameter<std::string>("superClusterAlgo");

  // determine which BasicCluster algo we are correcting for
  //And obtain forrection parameters form cfg file
  edm::ParameterSet fCorrPset;
  if (sCAlgo_str=="Hybrid") {
    sCAlgo_= reco::hybrid;
    fCorrPset = ps.getParameter<edm::ParameterSet>("hyb_fCorrPset"); 
  } else if (sCAlgo_str=="Island") {
    sCAlgo_= reco::island;
    fCorrPset = ps.getParameter<edm::ParameterSet>("isl_fCorrPset");
  } else if (sCAlgo_str=="DynamicHybrid") {
    sCAlgo_ = reco::dynamicHybrid;
    fCorrPset = ps.getParameter<edm::ParameterSet>("dyn_fCorrPset"); 
  } else if (sCAlgo_str=="FixedMatrix") {
    sCAlgo_ = reco::fixedMatrix;
    fCorrPset = ps.getParameter<edm::ParameterSet>("fix_fCorrPset");
  } else {
    edm::LogError("EgammaSCCorrectionMakerError") 
      << "Error! SuperClusterAlgo in config file must be Hybrid or Island: " 
      << sCAlgo_str << "  Using Hybrid by default";
    sCAlgo_=reco::hybrid;
  }
  
  // set correction algo parameters
  applyEnergyCorrection_ = ps.getParameter<bool>("applyEnergyCorrection");
  sigmaElectronicNoise_ =  ps.getParameter<double>("sigmaElectronicNoise");

  etThresh_ =  ps.getParameter<double>("etThresh");

  // set the producer parameters
  outputCollection_ = ps.getParameter<std::string>("corectedSuperClusterCollection");
  produces<reco::SuperClusterCollection>(outputCollection_);

  // instanciate the correction algo object
  energyCorrector_ = new EgammaSCEnergyCorrectionAlgo(sigmaElectronicNoise_, sCAlgo_, fCorrPset, verbosity_);
}

EgammaSCCorrectionMaker::~EgammaSCCorrectionMaker()
{
  delete energyCorrector_;
}

void
EgammaSCCorrectionMaker::produce(edm::Event& evt, const edm::EventSetup& es)
{
  using namespace edm;

  // get the collection geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry *geometry_p;

  std::string rHInputCollection = rHInputProducer_.instance();
  if(rHInputCollection == "EcalRecHitsEB") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  } else if(rHInputCollection == "EcalRecHitsEE") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  } else if(rHInputCollection == "EcalRecHitsPS") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  } else throw(std::runtime_error("\n\nSCCorrectionMaker encountered invalied ecalhitcollection type.\n\n"));
  
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
      reco::SuperCluster newClus;
      if(applyEnergyCorrection_) {
        newClus = energyCorrector_->applyCorrection(*aClus, *hitCollection, sCAlgo_, geometry_p);
      }

      if(newClus.energy()*sin(newClus.position().theta())>etThresh_) {
	//and corrected energy of SC before placing SCs in collection
	//std::cout << " Check 1 " << "\n"
	//	  << " Parameters of corrected SCs " << "\n"
	//	  << " energy = " << newClus.energy() <<"\n"
	//	  << " pw = " << newClus.phiWidth() << "\n"
	//	  << " ew = " << newClus.etaWidth() << std::endl;

	corrClusters->push_back(newClus);
      }
    }
  // Put collection of corrected SuperClusters into the event
  evt.put(corrClusters, outputCollection_);   
  
}

