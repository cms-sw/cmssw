#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoParticleFlow/PFProducer/test/PFSuperClusterReader.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

PFSuperClusterReader::PFSuperClusterReader(const edm::ParameterSet& iConfig)
{
  inputTagGSFTracks_  = iConfig.getParameter<edm::InputTag>("GSFTracks");  
  inputTagValueMapSC_  = iConfig.getParameter<edm::InputTag>("SuperClusterRefMap");  
  inputTagValueMapMVA_  = iConfig.getParameter<edm::InputTag>("MVAMap");  
}

PFSuperClusterReader::~PFSuperClusterReader(){;}

void PFSuperClusterReader::beginRun(edm::Run const&, edm::EventSetup const& ){;}

void PFSuperClusterReader::analyze(const edm::Event & iEvent,const edm::EventSetup & c)
{
  edm::Handle<reco::GsfTrackCollection> gsfTracksH;
  bool found=iEvent.getByLabel(inputTagGSFTracks_,gsfTracksH);
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get GsfTracks: "
       <<inputTagGSFTracks_<<std::endl;
    edm::LogError("PFSuperClusterReader")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }  
  
  edm::Handle<edm::ValueMap<reco::SuperClusterRef> > pfClusterTracksH;
  found = iEvent.getByLabel(inputTagValueMapSC_,pfClusterTracksH); 
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get SuperClusterRef Map: "
       <<inputTagValueMapSC_<<std::endl;
    edm::LogError("PFSuperClusterReader")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }  

  edm::Handle<edm::ValueMap<float> > pfMVAH;
  found = iEvent.getByLabel(inputTagValueMapMVA_,pfMVAH); 
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get MVA Map: "
       <<inputTagValueMapSC_<<std::endl;
    edm::LogError("PFSuperClusterReader")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }  

  const edm::ValueMap<reco::SuperClusterRef> & mySCValueMap=*pfClusterTracksH;
  const edm::ValueMap<float> & myMVAValueMap=*pfMVAH;

  unsigned ngsf=gsfTracksH->size();
  for(unsigned igsf=0;igsf<ngsf;++igsf)
    {
      reco::GsfTrackRef theTrackRef(gsfTracksH, igsf);
      if(mySCValueMap[theTrackRef].isNull()) continue;
      const reco::SuperCluster & mySuperCluster(*(mySCValueMap[theTrackRef]));
      float mva=myMVAValueMap[theTrackRef];
      std::cout << " Super Cluster energy " << mySuperCluster.energy() << std::endl;
      std::cout << " Super Cluster seed energy " << mySuperCluster.seed()->energy() << std::endl;
      std::cout << " Preshower contribution " << mySuperCluster.preshowerEnergy() << std::endl;
      std::cout << " MVA value " << mva << std::endl;
      std::cout << " List of basic clusters " << std::endl;
      reco::basicCluster_iterator it=mySuperCluster.clustersBegin();
      reco::basicCluster_iterator it_end=mySuperCluster.clustersEnd();
      for(;it!=it_end;++it)
	{
	  std::cout << " Basic cluster " << (*it)->energy() << std::endl ;
	}
      std::cout << std::endl;
    }
}

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(PFSuperClusterReader);


