#include "RecoEcal/EgammaClusterProducers/interface/EgammaSCCorrectionMaker.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Handle.h"

#include <string>

EgammaSCCorrectionMaker::EgammaSCCorrectionMaker(const edm::ParameterSet& ps)
{
  rHInputProducer_ = ps.getParameter<std::string>("recHitProducer");
  rHInputCollection_ = ps.getParameter<std::string>("recHitCollection");	
  sCInputProducer_ = ps.getParameter<std::string>("rawSuperClusterProducer");
  sCInputCollection_ = ps.getParameter<std::string>("rawSuperClusterCollection");
  applyEnergyCorrection_ = ps.getParameter<bool>("applyEnergyCorrection");
  sigmaElectronicNoise_ =  ps.getParameter<double>("sigmaElectronicNoise");

  outputCollection_ = ps.getParameter<std::string>("corectedSuperClusterCollection");
  produces<reco::SuperClusterCollection>(outputCollection_);

  energyCorrector_ = new EgammaSCEnergyCorrectionAlgo(sigmaElectronicNoise_);
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
    edm::LogError("EgammaSCCorrectionMakerError") << "Error! can't get the rawSuperClusters " << sCInputCollection_.c_str() ;
  }    
  
  // Get the RecHits from the event
  Handle<EcalRecHitCollection> pRecHits;
  try { 
    evt.getByLabel(rHInputProducer_, rHInputCollection_, pRecHits);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EgammaSCCorrectionMakerError") << "Error! can't get the RecHits " << rHInputCollection_.c_str() ;
  }    
  
  // Create a pointer to the RecHits and raw SuperClusters
  const EcalRecHitCollection *hitCollection = pRecHits.product();
  const reco::SuperClusterCollection *rawClusters = pRawSuperClusters.product();
   
  edm::LogInfo("EgammaSCCorrectionMakerInfo") << "Total # raw SCs " << rawClusters->size();   
   
  // Define a collection of corrected SuperClusters to put back into the event
  std::auto_ptr<reco::SuperClusterCollection> corrClusters(new reco::SuperClusterCollection);

  //  Loop over raw clusters and make corrected ones
  reco::SuperClusterCollection::const_iterator aClus;
  for(aClus = rawClusters->begin(); aClus != rawClusters->end(); aClus++)
    {
  	  reco::SuperCluster newClus;
      if(applyEnergyCorrection_) {
        newClus = energyCorrector_->applyCorrection(*aClus, *hitCollection);
      }
      corrClusters->push_back(newClus);
  }
 
  edm::LogInfo("EgammaSCCorrectionMakerInfo") << "Total # corrected SCs " << corrClusters->size(); 
 
  // Put collection of corrected SuperClusters into the event
  evt.put(corrClusters, outputCollection_);   
  edm::LogInfo("EgammaSCCorrectionMakerInfo") << "Put corrected SCs into the event!";
    
}

