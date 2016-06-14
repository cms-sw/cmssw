#ifndef _HGCROIAnalyzer_h_
#define _HGCROIAnalyzer_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "RecoLocalCalo/HGCalRecHitDump/interface/SlimmedRecHit.h"
#include "RecoLocalCalo/HGCalRecHitDump/interface/SlimmedROI.h"
#include "RecoLocalCalo/HGCalRecHitDump/interface/SlimmedVertex.h"
#include "RecoLocalCalo/HGCalRecHitDump/interface/SlimmedCluster.h"

#include "TTree.h"
#include "TVector3.h"
#include "TLorentzVector.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <unordered_map>

/**
   @class HGCROIAnalyzer
   @author P. Silva (CERN)
   @author L. Gray  (FNAL)
*/

class HGCROIAnalyzer : public edm::stream::EDAnalyzer<>
{  
 public:
  
  explicit HGCROIAnalyzer( const edm::ParameterSet& );
  ~HGCROIAnalyzer();
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

 private:
  
  void slimRecHits(const edm::Event &iEvent, const edm::EventSetup &iSetup);
  void doMCJetMatching(edm::Handle<std::vector<reco::PFJet> > &pfJets,
		       edm::Handle<reco::GenJetCollection> &genJets,
		       edm::Handle<edm::View<reco::Candidate> > &genParticles,
		       std::unordered_map<uint32_t,uint32_t> &reco2genJet,
		       std::unordered_map<uint32_t,uint32_t> &genJet2Parton,
		       std::unordered_map<uint32_t,uint32_t> &genJet2Stable);
  void doMCJetMatching(edm::Handle<reco::SuperClusterCollection> &superClusters,
		       edm::Handle<reco::GenJetCollection> &genJets,
		       edm::Handle<edm::View<reco::Candidate> > &genParticles,
		       std::unordered_map<uint32_t,uint32_t> &reco2genJet,
		       std::unordered_map<uint32_t,uint32_t> &genJet2Parton,
		       std::unordered_map<uint32_t,uint32_t> &genJet2Stable);

  virtual void endJob() ;

  TTree *tree_;
  Int_t run_,event_,lumi_;
  std::vector<SlimmedRecHit> *slimmedRecHits_;
  std::vector<SlimmedCluster> *slimmedClusters_;
  std::vector<SlimmedROI> *slimmedROIs_;
  std::vector<SlimmedVertex> *slimmedVertices_;
  TLorentzVector *genVertex_;
  
  bool useSuperClustersAsROIs_,useStatus3ForGenVertex_, useStatus3AsROIs_;
  edm::EDGetTokenT<edm::PCaloHitContainer> eeSimHitsSource_, hefSimHitsSource_;
  edm::EDGetTokenT<HGCRecHitCollection> eeRecHitsSource_, hefRecHitsSource_;
  edm::EDGetTokenT<std::vector<SimTrack> > g4TracksSource_;
  edm::EDGetTokenT<std::vector<SimVertex> > g4VerticesSource_;
  edm::EDGetTokenT<reco::VertexCollection> recoVertexSource_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > genSource_;
  edm::EDGetTokenT<reco::GenParticleCollection> genCandsFromSimTracksSource_;
  edm::EDGetTokenT<reco::GenJetCollection> genJetsSource_;
  edm::EDGetTokenT<std::vector<int> > genBarcodesSource_;
  edm::EDGetTokenT<std::vector<reco::PFJet> > pfJetsSource_;
  edm::EDGetTokenT<reco::SuperClusterCollection> superClustersSource_;
  edm::EDGetTokenT<edm::HepMCProduct> hepmceventSource_;
};
 

#endif
