#include "RecoParticleFlow/PFProducer/plugins/PFBlockProducer.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h" 
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <set>

using namespace std;
using namespace edm;

PFBlockProducer::PFBlockProducer(const edm::ParameterSet& iConfig) {
  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  bool debug_ = 
    iConfig.getUntrackedParameter<bool>("debug",false);  
  pfBlockAlgo_.setDebug(debug_);  
      
  edm::ConsumesCollector coll = consumesCollector();
  const std::vector<edm::ParameterSet>& importers
    = iConfig.getParameterSetVector("elementImporters");      
  pfBlockAlgo_.setImporters(importers,coll);

  const std::vector<edm::ParameterSet>& linkdefs 
    = iConfig.getParameterSetVector("linkDefinitions");
  pfBlockAlgo_.setLinkers(linkdefs);  
  
  produces<reco::PFBlockCollection>();

}



PFBlockProducer::~PFBlockProducer() { }



void 
PFBlockProducer::produce(Event& iEvent, 
			 const EventSetup& iSetup) {
    
  pfBlockAlgo_.buildElements(iEvent);

  pfBlockAlgo_.findBlocks();
  
  if(verbose_) {
    ostringstream  str;
    str<<pfBlockAlgo_<<endl;
    LogInfo("PFBlockProducer") << str.str()<<endl;
  }    

  auto_ptr< reco::PFBlockCollection > 
    pOutputBlockCollection( pfBlockAlgo_.transferBlocks() ); 
  
  iEvent.put(pOutputBlockCollection);

  LogDebug("PFBlockProducer")<<"STOP event: "<<iEvent.id().event()
			     <<" in run "<<iEvent.id().run()<<endl;
}
