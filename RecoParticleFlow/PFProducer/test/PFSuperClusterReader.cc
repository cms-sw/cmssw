#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
    : inputTagGSFTracks_(iConfig.getParameter<edm::InputTag>("GSFTracks")),
      inputTagValueMapSC_(iConfig.getParameter<edm::InputTag>("SuperClusterRefMap")),
      inputTagValueMapMVA_(iConfig.getParameter<edm::InputTag>("MVAMap")),
      inputTagPFCandidates_(iConfig.getParameter<edm::InputTag>("PFCandidate")),
      trackToken_(consumes<reco::GsfTrackCollection>(inputTagGSFTracks_)),
      pfCandToken_(consumes<reco::PFCandidateCollection>(inputTagPFCandidates_)),
      pfClusToken_(consumes<edm::ValueMap<reco::SuperClusterRef> >(inputTagValueMapSC_)),
      pfMapToken_(consumes<edm::ValueMap<float> >(inputTagValueMapMVA_)) {}

void PFSuperClusterReader::analyze(const edm::Event& iEvent, const edm::EventSetup& c) {
  const edm::Handle<reco::GsfTrackCollection>& gsfTracksH = iEvent.getHandle(trackToken_);
  if (!gsfTracksH.isValid()) {
    std::ostringstream err;
    err << " cannot get GsfTracks: " << inputTagGSFTracks_ << std::endl;
    edm::LogError("PFSuperClusterReader") << err.str();
    throw cms::Exception("MissingProduct", err.str());
  }

  const edm::Handle<reco::PFCandidateCollection>& pfCandidatesH = iEvent.getHandle(pfCandToken_);
  if (!pfCandidatesH.isValid()) {
    std::ostringstream err;
    err << " cannot get PFCandidates: " << inputTagPFCandidates_ << std::endl;
    edm::LogError("PFSuperClusterReader") << err.str();
    throw cms::Exception("MissingProduct", err.str());
  }

  const edm::Handle<edm::ValueMap<reco::SuperClusterRef> >& pfClusterTracksH = iEvent.getHandle(pfClusToken_);
  if (!pfClusterTracksH.isValid()) {
    std::ostringstream err;
    err << " cannot get SuperClusterRef Map: " << inputTagValueMapSC_ << std::endl;
    edm::LogError("PFSuperClusterReader") << err.str();
    throw cms::Exception("MissingProduct", err.str());
  }

  const edm::Handle<edm::ValueMap<float> >& pfMVAH = iEvent.getHandle(pfMapToken_);
  if (!pfMVAH.isValid()) {
    std::ostringstream err;
    err << " cannot get MVA Map: " << inputTagValueMapSC_ << std::endl;
    edm::LogError("PFSuperClusterReader") << err.str();
    throw cms::Exception("MissingProduct", err.str());
  }

  const edm::ValueMap<reco::SuperClusterRef>& mySCValueMap = *pfClusterTracksH;
  const edm::ValueMap<float>& myMVAValueMap = *pfMVAH;

  unsigned ngsf = gsfTracksH->size();
  for (unsigned igsf = 0; igsf < ngsf; ++igsf) {
    reco::GsfTrackRef theTrackRef(gsfTracksH, igsf);
    if (mySCValueMap[theTrackRef].isNull()) {
      continue;
    }
    const reco::SuperCluster& mySuperCluster(*(mySCValueMap[theTrackRef]));
    float mva = myMVAValueMap[theTrackRef];
    float eta = mySuperCluster.position().eta();
    float et = mySuperCluster.energy() * sin(mySuperCluster.position().theta());
    edm::LogVerbatim("PFSuperClusterReader")
        << " Super Cluster energy, eta, Et , EtaWidth, PhiWidth " << mySuperCluster.energy() << " " << eta << " " << et
        << " " << mySuperCluster.etaWidth() << " " << mySuperCluster.phiWidth();

    const reco::PFCandidate* myPFCandidate = findPFCandidate(pfCandidatesH.product(), theTrackRef);

    if (mySuperCluster.seed().isNull()) {
      continue;
    }
    edm::LogVerbatim("PFSuperClusterReader") << " Super Cluster seed energy " << mySuperCluster.seed()->energy();
    edm::LogVerbatim("PFSuperClusterReader") << " Preshower contribution " << mySuperCluster.preshowerEnergy();
    edm::LogVerbatim("PFSuperClusterReader") << " MVA value " << mva;
    edm::LogVerbatim("PFSuperClusterReader") << " List of basic clusters ";
    reco::CaloCluster_iterator it = mySuperCluster.clustersBegin();
    reco::CaloCluster_iterator it_end = mySuperCluster.clustersEnd();
    float etotbasic = 0;
    for (; it != it_end; ++it) {
      edm::LogVerbatim("PFSuperClusterReader") << " Basic cluster " << (*it)->energy();
      etotbasic += (*it)->energy();
    }
    it = mySuperCluster.preshowerClustersBegin();
    it_end = mySuperCluster.preshowerClustersEnd();
    for (; it != it_end; ++it) {
      edm::LogVerbatim("PFSuperClusterReader") << " Preshower cluster " << (*it)->energy();
    }

    edm::LogVerbatim("PFSuperClusterReader") << " Comparison with PFCandidate : Energy " << myPFCandidate->ecalEnergy()
                                             << " SC : " << mySuperCluster.energy();
    std::ostringstream st1;
    st1 << " Sum of Basic clusters :" << etotbasic;
    st1 << " Calibrated preshower energy : " << mySuperCluster.preshowerEnergy();
    etotbasic += mySuperCluster.preshowerEnergy();
    st1 << " Basic Clusters + PS :" << etotbasic;
    edm::LogVerbatim("PFSuperClusterReader") << st1.str();
    edm::LogVerbatim("PFSuperClusterReader") << " Summary " << mySuperCluster.energy() << " " << eta << " " << et << " "
                                             << mySuperCluster.preshowerEnergy() << " " << mva << std::endl;
  }
}

const reco::PFCandidate* PFSuperClusterReader::findPFCandidate(const reco::PFCandidateCollection* coll,
                                                               const reco::GsfTrackRef& ref) {
  const reco::PFCandidate* result = 0;
  unsigned ncand = coll->size();
  for (unsigned icand = 0; icand < ncand; ++icand) {
    if (!(*coll)[icand].gsfTrackRef().isNull() && (*coll)[icand].gsfTrackRef() == ref) {
      result = &((*coll)[icand]);
      return result;
    }
  }
  return result;
}

DEFINE_FWK_MODULE(PFSuperClusterReader);
