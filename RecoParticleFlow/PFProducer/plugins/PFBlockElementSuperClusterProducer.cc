#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoParticleFlow/PFProducer/plugins/PFBlockElementSuperClusterProducer.h"
#include "RecoParticleFlow/PFProducer/interface/PFPhotonSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

PFBlockElementSuperClusterProducer::PFBlockElementSuperClusterProducer(const edm::ParameterSet & iConfig) {
  inputTagSuperClusters_ 
    = iConfig.getParameter<std::vector<edm::InputTag> >("ECALSuperClusters");
  
  outputName_ = iConfig.getParameter<std::string>("PFBESuperClusters");

  produces<reco::PFBlockElementSuperClusterCollection>(outputName_); 
}

PFBlockElementSuperClusterProducer::~PFBlockElementSuperClusterProducer() {}

void PFBlockElementSuperClusterProducer::beginRun(edm::Run& run,const edm::EventSetup & es) {}

void PFBlockElementSuperClusterProducer::produce(edm::Event& iEvent,  
				    const edm::EventSetup& iSetup) { 
  
  std::auto_ptr<reco::PFBlockElementSuperClusterCollection> 
    pfbeSuperClusters_p(new reco::PFBlockElementSuperClusterCollection);
  

  for(unsigned icol=0; icol<inputTagSuperClusters_.size() ; ++icol) {
    edm::Handle<reco::SuperClusterCollection> scH;
    bool found=iEvent.getByLabel(inputTagSuperClusters_[icol],scH);
    if(!found) {
      std::ostringstream  err;
      err<<" cannot get SuperClusters: "
	 <<inputTagSuperClusters_[icol]<<std::endl;
      edm::LogError("PFBlockElementSuperClusterProducer")<<err.str();
      throw cms::Exception( "MissingProduct", err.str());      
    }

    unsigned nsc=scH->size();

    for(unsigned isc=0;isc<nsc;++isc) {
      reco::SuperClusterRef theRef(scH,isc);
      
      // create the object
      reco::PFBlockElementSuperCluster myPFBE(theRef);

      // to use the standard tools, create a fake photon object
      reco::Candidate::LorentzVector p4(theRef->position().X(),
					theRef->position().Y(),
					theRef->position().Z(),
					theRef->energy());
      reco::PFPhotonSuperCluster myFakePhoton(p4,theRef);



      myPFBE.setTrackIso(0.);
      myPFBE.setEcalIso(0.);
      myPFBE.setHcalIso(0.);
      myPFBE.setHoE(0.);
      pfbeSuperClusters_p->push_back(myPFBE);
    }    
  }    
  std::cout << "Size " << pfbeSuperClusters_p->size() << std::endl;
  iEvent.put(pfbeSuperClusters_p,outputName_);
}

