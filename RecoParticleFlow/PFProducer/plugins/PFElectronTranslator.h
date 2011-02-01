#ifndef RecoParticleFlow_PFProducer_PFElectronTranslator_H
#define RecoParticleFlow_PFProducer_PFElectronTranslator_H
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include <iostream>
#include <string>
#include <map>



class PFElectronTranslator : public edm::EDProducer
{
 public:
  explicit PFElectronTranslator(const edm::ParameterSet&);
  ~PFElectronTranslator();
  
  virtual void produce(edm::Event &, const edm::EventSetup&);
  virtual void beginRun(edm::Run & run,const edm::EventSetup & c);

 private:
  // to retrieve the collection from the event
  bool fetchCandidateCollection(edm::Handle<reco::PFCandidateCollection>& c, 
				const edm::InputTag& tag, 
				const edm::Event& iEvent) const;
  // to retrieve the collection from the event
  void fetchGsfCollection(edm::Handle<reco::GsfTrackCollection>& c, 
			  const edm::InputTag& tag, 
			  const edm::Event& iEvent) const ;

  // makes a basic cluster from PFBlockElement and add it to the collection ; the corrected energy is taken
  // from the PFCandidate
  void createBasicCluster(const reco::PFBlockElement & ,  reco::BasicClusterCollection & basicClusters,
			  std::vector<const reco::PFCluster *> &,
			  const reco::PFCandidate & coCandidate) const;
  // makes a preshower cluster from of PFBlockElement and add it to the collection
  void createPreshowerCluster(const reco::PFBlockElement & PFBE, 
			      reco::PreshowerClusterCollection& preshowerClusters,
			      unsigned plane) const;

  // make a super cluster from its ingredients and add it to the collection
  void createSuperClusters(const reco::PFCandidateCollection &,
			  reco::SuperClusterCollection &superClusters) const;

  // create the basic cluster Ptr
  void createBasicClusterPtrs(const edm::OrphanHandle<reco::BasicClusterCollection> & basicClustersHandle );

  // create the preshower cluster Refs
  void createPreshowerClusterPtrs(const edm::OrphanHandle<reco::PreshowerClusterCollection> & preshowerClustersHandle );

  // create the super cluster Refs
  void createSuperClusterGsfMapRefs(const edm::OrphanHandle<reco::SuperClusterCollection> & superClustersHandle );

  // The following methods are used to fill the value maps
  void fillMVAValueMap(edm::Event& iEvent, edm::ValueMap<float>::Filler & filler) const;
  void fillValueMap(edm::Event& iEvent, edm::ValueMap<float>::Filler & filler) const;
  void fillSCRefValueMap(edm::Event& iEvent, 
			 edm::ValueMap<reco::SuperClusterRef>::Filler & filler) const;

  const reco::PFCandidate & correspondingDaughterCandidate(const reco::PFCandidate & cand, const reco::PFBlockElement & pfbe) const;
 private:
  edm::InputTag inputTagPFCandidates_;
  edm::InputTag inputTagGSFTracks_;
  std::string PFBasicClusterCollection_;
  std::string PFPreshowerClusterCollection_;
  std::string PFSuperClusterCollection_;
  std::string PFMVAValueMap_;
  std::string PFSCValueMap_;
  double MVACut_;

  // The following vectors correspond to a GSF track, but the order is not 
  // the order of the tracks in the GSF track collection
  std::vector<reco::GsfTrackRef> GsfTrackRef_;
  // the collection of basic clusters associated to a GSF track
  std::vector<reco::BasicClusterCollection> basicClusters_;
  // the correcsponding PFCluster ref
  std::vector<std::vector<const reco::PFCluster *> > pfClusters_;
  // the collection of preshower clusters associated to a GSF track
  std::vector<reco::PreshowerClusterCollection> preshowerClusters_;
  // the super cluster collection (actually only one) associated to a GSF trck
  std::vector<reco::SuperClusterCollection> superClusters_;
  // the references to the basic clusters associated to a GSF track
  std::vector<reco::CaloClusterPtrVector> basicClusterPtr_;
  // the references to the basic clusters associated to a GSF track
  std::vector<reco::CaloClusterPtrVector> preshowerClusterPtr_;
  // keep track of the index of the PF Candidate
  std::vector<int> gsfPFCandidateIndex_;
  // maps to ease the creation of the Value Maps 
  std::map<reco::GsfTrackRef,reco::SuperClusterRef> scMap_;
  std::map<reco::GsfTrackRef,float> gsfMvaMap_;
  
  bool emptyIsOk_;

};
#endif
