#include "RecoEcal/EgammaClusterProducers/interface/EgammaSCCorrectionMaker.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include <string>

EgammaSCCorrectionMaker::EgammaSCCorrectionMaker(const edm::ParameterSet& ps)
{

  // The verbosity level
  std::string debugString = ps.getParameter<std::string>("VerbosityLevel");
  if      (debugString == "DEBUG")   verbosity_ = EgammaSCEnergyCorrectionAlgo::pDEBUG;
  else if (debugString == "INFO")    verbosity_ = EgammaSCEnergyCorrectionAlgo::pINFO;
  else                               verbosity_ = EgammaSCEnergyCorrectionAlgo::pERROR;

  // the input producers
  rHInputProducer_ = ps.getParameter<std::string>("recHitProducer");
  rHInputCollection_ = ps.getParameter<std::string>("recHitCollection");	
  sCInputProducer_ = ps.getParameter<std::string>("rawSuperClusterProducer");
  sCInputCollection_ = ps.getParameter<std::string>("rawSuperClusterCollection");
  std::string sCAlgo_str = ps.getParameter<std::string>("superClusterAlgo");

  // determine which BasicCluster algo we are correcting for
  if (sCAlgo_str=="Hybrid") {
    sCAlgo_= reco::hybrid;
  } else if (sCAlgo_str=="Island") {
    sCAlgo_= reco::island;
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
  energyCorrector_ = new EgammaSCEnergyCorrectionAlgo(sigmaElectronicNoise_, verbosity_);
}

EgammaSCCorrectionMaker::~EgammaSCCorrectionMaker()
{
  delete energyCorrector_;
}

void
EgammaSCCorrectionMaker::produce(edm::Event& evt, const edm::EventSetup& es)
{
  using namespace edm;
   
  // Get raw SuperClusters from the event    
  Handle<reco::SuperClusterCollection> pRawSuperClusters;
  try { 
    evt.getByLabel(sCInputProducer_, sCInputCollection_, pRawSuperClusters);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EgammaSCCorrectionMakerError") 
      << "Error! can't get the rawSuperClusters " 
      << sCInputCollection_.c_str() ;
  }    
  
  // Get the RecHits from the event
  Handle<EcalRecHitCollection> pRecHits;
  try { 
    evt.getByLabel(rHInputProducer_, rHInputCollection_, pRecHits);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EgammaSCCorrectionMakerError") 
      << "Error! can't get the RecHits " 
      << rHInputCollection_.c_str() ;
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
        newClus = energyCorrector_->applyCorrection(*aClus, *hitCollection, sCAlgo_);
      }

      if(newClus.energy()*sin(newClus.position().theta())>etThresh_)
	corrClusters->push_back(newClus);
    }
 
  // Put collection of corrected SuperClusters into the event
  evt.put(corrClusters, outputCollection_);   
  
}

