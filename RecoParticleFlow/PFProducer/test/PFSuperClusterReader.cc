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
  inputTagGSFTracks_    = iConfig.getParameter<edm::InputTag>("GSFTracks");  
  inputTagValueMapSC_   = iConfig.getParameter<edm::InputTag>("SuperClusterRefMap");  
  inputTagValueMapMVA_  = iConfig.getParameter<edm::InputTag>("MVAMap");  
  inputTagPFCandidates_ = iConfig.getParameter<edm::InputTag>("PFCandidate");
}

PFSuperClusterReader::~PFSuperClusterReader(){;}

void 
PFSuperClusterReader::beginRun(edm::Run const&, edm::EventSetup const& ){;}

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

  edm::Handle<reco::PFCandidateCollection> pfCandidatesH;
  found=iEvent.getByLabel(inputTagPFCandidates_,pfCandidatesH);
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get PFCandidates: "
       <<inputTagPFCandidates_<<std::endl;
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
      if(mySCValueMap[theTrackRef].isNull()) 
	{
	  continue;
	}
      const reco::SuperCluster & mySuperCluster(*(mySCValueMap[theTrackRef]));
      float mva=myMVAValueMap[theTrackRef];
      float eta=mySuperCluster.position().eta();
      float et=mySuperCluster.energy()*sin(mySuperCluster.position().theta());
      std::cout << " Super Cluster energy, eta, Et , EtaWidth, PhiWidth " << mySuperCluster.energy() << " " ;
      std::cout <<  eta << " " << et << " " << mySuperCluster.etaWidth() << " " << mySuperCluster.phiWidth() << std::endl;
      
      const reco::PFCandidate * myPFCandidate = findPFCandidate(pfCandidatesH.product(),theTrackRef);

      if(mySuperCluster.seed().isNull())
	{
	  continue;
	}
      std::cout << " Super Cluster seed energy " << mySuperCluster.seed()->energy() << std::endl;
      std::cout << " Preshower contribution " << mySuperCluster.preshowerEnergy() << std::endl;
      std::cout << " MVA value " << mva << std::endl;
      std::cout << " List of basic clusters " << std::endl;
      reco::CaloCluster_iterator it=mySuperCluster.clustersBegin();
      reco::CaloCluster_iterator it_end=mySuperCluster.clustersEnd();
      float etotbasic=0;
      for(;it!=it_end;++it)
	{
	  std::cout << " Basic cluster " << (*it)->energy() << std::endl ;
	  etotbasic += (*it)->energy();
	}
      it = mySuperCluster.preshowerClustersBegin();
      it_end = mySuperCluster.preshowerClustersEnd();
      for(;it!=it_end;++it)
	{
	  std::cout << " Preshower cluster " << (*it)->energy() << std::endl;
	}

      std::cout << " Comparison with PFCandidate : Energy " << myPFCandidate->ecalEnergy() << " SC : " << mySuperCluster.energy() << std::endl;
      std::cout << " Sum of Basic clusters :" << etotbasic ;
      std::cout << " Calibrated preshower energy : " << mySuperCluster.preshowerEnergy() ;
      etotbasic += mySuperCluster.preshowerEnergy();
      std::cout << " Basic Clusters + PS :" << etotbasic << std::endl;
      std::cout << " Summary " << mySuperCluster.energy() << " " << eta << " " << et ;
      std::cout << " " <<mySuperCluster.preshowerEnergy() << " " << mva << std::endl;
      std::cout << std::endl;
    }
}

const reco::PFCandidate * PFSuperClusterReader::findPFCandidate(const reco::PFCandidateCollection * coll,const reco::GsfTrackRef & ref)
{
  const reco::PFCandidate * result=0;
  unsigned ncand=coll->size();
  for(unsigned icand=0;icand<ncand;++icand)
    {
      if(!(*coll)[icand].gsfTrackRef().isNull() && (*coll)[icand].gsfTrackRef()==ref)
	{
	  result=&((*coll)[icand]);      
	  return result;
	}
    }
  return result;
}


DEFINE_FWK_MODULE(PFSuperClusterReader);


