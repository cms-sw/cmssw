// This producer assigns vertex times (with a specified resolution) to tracks.
// The times are produced as valuemaps associated to tracks, so the track dataformat doesn't
// need to be modified.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include <unordered_map>
#include <memory>

#include "CLHEP/Units/SystemOfUnits.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "FWCore/Utilities/interface/transform.h"

class SimPFProducer : public edm::global::EDProducer<> {
public:    
  SimPFProducer(const edm::ParameterSet&);
  ~SimPFProducer() { }
  
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
private:
  // parameters
  const double superClusterThreshold_;
  
  // inputs
  const edm::EDGetTokenT<edm::View<reco::Track> > tracks_;
  const edm::EDGetTokenT<edm::View<reco::Track> > gsfTracks_;
  const edm::EDGetTokenT<TrackingParticleCollection> trackingParticles_;
  const edm::EDGetTokenT<SimClusterCollection> simClustersTruth_;
  const edm::EDGetTokenT<CaloParticleCollection> caloParticles_;
  const edm::EDGetTokenT<std::vector<reco::PFCluster> > simClusters_;
  // tracking particle associators by order of preference
  const std::vector<edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> > associators_;     
};

DEFINE_FWK_MODULE(SimPFProducer);

namespace {
  
  template<typename T>
  uint64_t hashSimInfo(const T& simTruth,size_t i = 0) {
    uint64_t evtid = simTruth.eventId().rawId();
    uint64_t trackid = simTruth.g4Tracks()[i].trackId();
    return ( (evtid << 3) + 23401923 ) ^ trackid ;
  };
}

SimPFProducer::SimPFProducer(const edm::ParameterSet& conf) :
  superClusterThreshold_( conf.getParameter<double>("superClusterThreshold") ),
  tracks_(consumes<edm::View<reco::Track> >( conf.getParameter<edm::InputTag>("trackSrc") ) ),
  gsfTracks_(consumes<edm::View<reco::Track> >( conf.getParameter<edm::InputTag>("gsfTrackSrc") ) ),
  trackingParticles_(consumes<TrackingParticleCollection>( conf.getParameter<edm::InputTag>("trackingParticleSrc") ) ),
  simClustersTruth_(consumes<SimClusterCollection>( conf.getParameter<edm::InputTag>("simClusterTruthSrc") ) ),
  caloParticles_(consumes<CaloParticleCollection>( conf.getParameter<edm::InputTag>("caloParticlesSrc") ) ),
  simClusters_(consumes<std::vector<reco::PFCluster> >( conf.getParameter<edm::InputTag>("simClustersSrc") ) ),
  associators_( edm::vector_transform( conf.getParameter<std::vector<edm::InputTag> >("associators"), [this](const edm::InputTag& tag){ return this->consumes<reco::TrackToTrackingParticleAssociator>(tag); } ) )
{
  produces<reco::PFBlockCollection>();
  produces<reco::SuperClusterCollection>("perfect");
  produces<reco::PFCandidateCollection>();
}

void SimPFProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {  
  //get associators
  std::vector<edm::Handle<reco::TrackToTrackingParticleAssociator> > associators;
  for( const auto& token : associators_ ) {
    associators.emplace_back();
    auto& back = associators.back();
    evt.getByToken(token,back);
  }
  
  //get track collections
  edm::Handle<edm::View<reco::Track> > TrackCollectionH;
  evt.getByToken(tracks_, TrackCollectionH);
  const edm::View<reco::Track>& TrackCollection = *TrackCollectionH;

  /*
  edm::Handle<edm::View<reco::Track> > GsfTrackCollectionH;
  evt.getByToken(gsfTracks_, GsfTrackCollectionH);
  const edm::View<reco::Track>& GsfTrackCollection = *GsfTrackCollectionH;
  */
  
  //get tracking particle collections
  edm::Handle<TrackingParticleCollection>  TPCollectionH;
  evt.getByToken(trackingParticles_, TPCollectionH);
  //const TrackingParticleCollection&  TPCollection = *TPCollectionH;

  // grab phony clustering information
  edm::Handle<SimClusterCollection> SimClustersTruthH;
  evt.getByToken(simClustersTruth_,SimClustersTruthH);
  const SimClusterCollection& SimClustersTruth = *SimClustersTruthH;

  edm::Handle<CaloParticleCollection> CaloParticlesH;
  evt.getByToken(caloParticles_,CaloParticlesH);
  const CaloParticleCollection& CaloParticles = *CaloParticlesH;

  edm::Handle<std::vector<reco::PFCluster> > SimClustersH;
  evt.getByToken(simClusters_,SimClustersH);
  const std::vector<reco::PFCluster>& SimClusters = *SimClustersH;

  std::unordered_map<uint64_t,size_t> hashToSimCluster; 

  for( unsigned i = 0; i < SimClustersTruth.size(); ++i ) {
    const auto& simTruth = SimClustersTruth[i];
    hashToSimCluster[hashSimInfo(simTruth)] = i;
  }

  // associate the reco tracks / gsf Tracks
  std::vector<reco::RecoToSimCollection> associatedTracks, associatedTracksGsf;  
  for( auto associator : associators ) {
    associatedTracks.emplace_back(associator->associateRecoToSim(TrackCollectionH, TPCollectionH));
    //associatedTracksGsf.emplace_back(associator->associateRecoToSim(GsfTrackCollectionH, TPCollectionH));
  }

  // make blocks out of calo particles so we can have cluster references
  // likewise fill out superclusters
  std::unique_ptr<reco::SuperClusterCollection> superclusters(new reco::SuperClusterCollection);
  std::unique_ptr<reco::PFBlockCollection> blocks(new reco::PFBlockCollection);
  std::unordered_map<size_t,size_t> simCluster2Block;
  std::unordered_map<size_t,size_t> simCluster2BlockIndex;
  std::unordered_multimap<size_t,size_t> caloParticle2SimCluster;
  std::vector<int> caloParticle2SuperCluster;
  for( unsigned icp = 0; icp < CaloParticles.size(); ++icp ) {
    blocks->emplace_back();
    auto block = blocks->back();
    const auto& simclusters = CaloParticles[icp].simClusters();
    double pttot = 0.0;
    double etot  = 0.0;
    std::vector<size_t> good_simclusters;
    for( unsigned isc = 0; isc < simclusters.size() ; ++isc ) {
      auto simc = simclusters[isc];
      edm::Ref<std::vector<reco::PFCluster> > clusterRef(SimClustersH,simc.key());
      if( clusterRef->energy() > 0.0 ) {	
	good_simclusters.push_back(isc);
	etot += clusterRef->energy();
	pttot += clusterRef->pt();	
	std::unique_ptr<reco::PFBlockElementCluster> bec( new reco::PFBlockElementCluster(clusterRef,reco::PFBlockElement::HGCAL) );
	block.addElement(bec.get());
	simCluster2Block[simc.key()] = icp;
	simCluster2BlockIndex[simc.key()] = bec->index();
	caloParticle2SimCluster.emplace(icp,simc.key());
      }
    }
    auto pdgId = std::abs(CaloParticles[icp].particleId());
    caloParticle2SuperCluster.push_back(-1);
    if( (pdgId == 22 || pdgId == 11) && pttot > superClusterThreshold_ ) {
      caloParticle2SuperCluster[icp] = superclusters->size();
      math::XYZPoint seedpos; // take seed pos as supercluster point
      reco::CaloClusterPtr seed;
      reco::CaloClusterPtrVector clusters;
      for( auto idx : good_simclusters ) {
	edm::Ptr<reco::PFCluster> ptr(SimClustersH, simclusters[idx].key());
	clusters.push_back(ptr);
	if( seed.isNull() || seed->energy() < ptr->energy() ) {
	  seed = ptr;
	  seedpos = ptr->position();
	}
      }
      superclusters->emplace_back(etot,seedpos,seed,clusters);
    }
  }
  auto blocksHandle = evt.put(std::move(blocks));
  auto superClustersHandle = evt.put(std::move(superclusters),"perfect");
  
  // list tracks so we can mark them as used and/or fight over them
  std::vector<bool> usedTrack(TrackCollection.size(),false), 
                  //usedGsfTrack(GsfTrackCollection.size(),false), 
                    usedSimCluster(SimClusters.size(),false);

  std::unique_ptr<reco::PFCandidateCollection> candidates(new reco::PFCandidateCollection);
  // in good particle flow fashion, start from the tracks and go out
  for( unsigned itk = 0; itk < TrackCollection.size(); ++itk ) {
    auto tkRef  = TrackCollection.refAt(itk);
    std::cout << "Track: " << itk << ' ' << tkRef->pt() << ' ' << tkRef->eta() << ' ' << tkRef->phi() << std::endl;
    reco::RecoToSimCollection::const_iterator assoc_tps = associatedTracks.back().end();
    for( const auto& association : associatedTracks ) {
      assoc_tps = association.find(tkRef);
      if( assoc_tps != association.end() ) break;
    }
    if( assoc_tps == associatedTracks.back().end() ) continue;
    // assured now that we are matched to a set of tracks
    const auto& matches = assoc_tps->val;
    
    const auto absPdgId = std::abs(matches[0].first->pdgId());
    const auto charge = tkRef->charge();
    const auto three_mom = tkRef->momentum();
    constexpr double mpion2 = 0.13957*0.13957;
    double energy = std::sqrt(three_mom.mag2() + mpion2);
    math::XYZTLorentzVector trk_p4(three_mom.x(),three_mom.y(),three_mom.z(),energy);
    
    reco::PFCandidate::ParticleType part_type;

    switch( absPdgId ) {
    case 11:
      part_type = reco::PFCandidate::e;
      break;
    case 13:
      part_type = reco::PFCandidate::mu;
      break;
    default:
      part_type = reco::PFCandidate::h;
    }
    
    candidates->emplace_back(charge, trk_p4, part_type);
    auto& candidate = candidates->back();
    candidate.setTrackRef(tkRef.castTo<reco::TrackRef>());
    
    // bind to cluster if there is one and try to gather conversions, etc
    for( const auto& match : matches ) {      
      std::cout << "\tMatches: " << *(match.first) << std::endl;
      uint64_t hash = hashSimInfo(*(match.first));
      if( hashToSimCluster.count(hash) ) {	
	auto simcHash = hashToSimCluster[hash];
	auto& simc = SimClusters[simcHash];
	std::cout << "Matches SimCluster: " << simc.pt() << ' ' << simc.eta() << ' ' << simc.phi() << std::endl;
	if( !usedSimCluster[simcHash] ) {
	  std::cout << "\tSimCluster not used!" << std::endl;
	  size_t block    = simCluster2Block.find(simcHash)->second;
	  size_t blockIdx = simCluster2BlockIndex.find(simcHash)->second;
	  edm::Ref<reco::PFBlockCollection> blockRef(blocksHandle,block);
	  std::cout << "block size = " << blockRef->elements().size() << std::endl;
	  candidate.addElementInBlock(blockRef,blockIdx);
	  usedSimCluster[simcHash] = true;
	}
	if( absPdgId == 11 ) { // collect brems/conv. brems
	  std::cout << "caught an electron" << std::endl;
	  const auto& g4tracks = match.first->g4Tracks();
	  std::cout << g4tracks.size() << " associated tracks" << std::endl;
	  for( unsigned ig4 = 1; ig4 < g4tracks.size(); ++ig4 ) {
	    uint64_t subhash = hashSimInfo(*(match.first),ig4);
	    if( hashToSimCluster.count(subhash) ) {	
	      auto sub_simcHash = hashToSimCluster[subhash];
	      auto& sub_simc = SimClusters[sub_simcHash];
	      std::cout << "Matches SimCluster: " << sub_simc.pt() << ' ' << sub_simc.eta() << ' ' << sub_simc.phi() << std::endl;
	      if( !usedSimCluster[sub_simcHash] ) {
		std::cout << "\tsub-SumCluster not used!" << std::endl;
		size_t block    = simCluster2Block.find(sub_simcHash)->second;
		size_t blockIdx = simCluster2BlockIndex.find(sub_simcHash)->second;
		edm::Ref<reco::PFBlockCollection> blockRef(blocksHandle,block);
		candidate.addElementInBlock(blockRef,blockIdx);
		usedSimCluster[sub_simcHash] = true;
	      }
	    }
	  }
	}
      }
    }
    usedTrack[tkRef.key()] = true;    

    std::cout << candidate << std::endl;   
  }  
  
  evt.put(std::move(candidates));
}
