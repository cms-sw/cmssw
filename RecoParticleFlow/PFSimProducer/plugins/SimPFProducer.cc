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

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

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
  ~SimPFProducer() override { }
  
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
private:  
  // parameters
  const double superClusterThreshold_, neutralEMThreshold_, neutralHADThreshold_;
  const bool useTiming_;
  
  // inputs
  const edm::EDGetTokenT<edm::View<reco::PFRecTrack> > pfRecTracks_;
  const edm::EDGetTokenT<edm::View<reco::Track> > tracks_;
  const edm::EDGetTokenT<edm::View<reco::Track> > gsfTracks_;
  const edm::EDGetTokenT<reco::MuonCollection> muons_;  
  const edm::EDGetTokenT<edm::ValueMap<float>> srcTrackTime_, srcTrackTimeError_;
  const edm::EDGetTokenT<edm::ValueMap<float>> srcGsfTrackTime_, srcGsfTrackTimeError_;
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
  neutralEMThreshold_( conf.getParameter<double>("neutralEMThreshold") ),
  neutralHADThreshold_( conf.getParameter<double>("neutralHADThreshold") ),
  useTiming_(conf.existsAs<edm::InputTag>("trackTimeValueMap")),
  pfRecTracks_(consumes<edm::View<reco::PFRecTrack> >(conf.getParameter<edm::InputTag> ("pfRecTrackSrc"))),
  tracks_(consumes<edm::View<reco::Track> >( conf.getParameter<edm::InputTag>("trackSrc") ) ),
  gsfTracks_(consumes<edm::View<reco::Track> >( conf.getParameter<edm::InputTag>("gsfTrackSrc") ) ),
  muons_(consumes<reco::MuonCollection>(conf.getParameter<edm::InputTag>("muonSrc"))),
  srcTrackTime_(useTiming_ ? consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("trackTimeValueMap")) : edm::EDGetTokenT<edm::ValueMap<float>>()),
  srcTrackTimeError_(useTiming_ ? consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("trackTimeErrorMap")) : edm::EDGetTokenT<edm::ValueMap<float>>()),
  srcGsfTrackTime_(useTiming_ ? consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("gsfTrackTimeValueMap")) : edm::EDGetTokenT<edm::ValueMap<float>>()),
  srcGsfTrackTimeError_(useTiming_ ? consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("gsfTrackTimeErrorMap")) : edm::EDGetTokenT<edm::ValueMap<float>>()),
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
  
  //get PFRecTrack
  edm::Handle<edm::View<reco::PFRecTrack> > PFTrackCollectionH;
  evt.getByToken(pfRecTracks_,PFTrackCollectionH);
  const edm::View<reco::PFRecTrack> PFTrackCollection = *PFTrackCollectionH;
  std::unordered_set<unsigned> PFTrackToGeneralTrack;
  for( unsigned i = 0; i < PFTrackCollection.size(); ++i ) {
    const auto ptr = PFTrackCollection.ptrAt(i);
    PFTrackToGeneralTrack.insert(ptr->trackRef().key());
  }
  
  //get track collections
  edm::Handle<edm::View<reco::Track> > TrackCollectionH;
  evt.getByToken(tracks_, TrackCollectionH);
  const edm::View<reco::Track>& TrackCollection = *TrackCollectionH;

  edm::Handle<reco::MuonCollection> muons;
  evt.getByToken(muons_,muons);
  std::unordered_set<unsigned> MuonTrackToGeneralTrack;
  for (auto const& mu : *muons.product()){
    reco::TrackRef muTrkRef = mu.track();
    if (muTrkRef.isNonnull())
      MuonTrackToGeneralTrack.insert(muTrkRef.key());
  }

  // get timing, if enabled
  edm::Handle<edm::ValueMap<float>> trackTimeH, trackTimeErrH, gsfTrackTimeH, gsfTrackTimeErrH;
  if (useTiming_) {
    evt.getByToken(srcTrackTime_, trackTimeH);
    evt.getByToken(srcTrackTimeError_, trackTimeErrH);
    evt.getByToken(srcGsfTrackTime_, gsfTrackTimeH);
    evt.getByToken(srcGsfTrackTimeError_, gsfTrackTimeErrH);
  }
 
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
  auto superclusters = std::make_unique<reco::SuperClusterCollection>();
  auto blocks = std::make_unique<reco::PFBlockCollection>();
  std::unordered_map<size_t,size_t> simCluster2Block;
  std::unordered_map<size_t,size_t> simCluster2BlockIndex;
  std::unordered_multimap<size_t,size_t> caloParticle2SimCluster;
  std::vector<int> caloParticle2SuperCluster;
  for( unsigned icp = 0; icp < CaloParticles.size(); ++icp ) {
    blocks->emplace_back();
    auto& block = blocks->back();
    const auto& simclusters = CaloParticles[icp].simClusters();
    double pttot = 0.0;
    double etot  = 0.0;
    std::vector<size_t> good_simclusters;
    for( unsigned isc = 0; isc < simclusters.size() ; ++isc ) {
      auto simc = simclusters[isc];
      auto pdgId = std::abs(simc->pdgId());
      edm::Ref<std::vector<reco::PFCluster> > clusterRef(SimClustersH,simc.key());
      if( ( (pdgId == 22 || pdgId == 11) &&  clusterRef->energy() >  neutralEMThreshold_) ||
	  clusterRef->energy() > neutralHADThreshold_ ) {	
	good_simclusters.push_back(isc);
	etot += clusterRef->energy();
	pttot += clusterRef->pt();	
	auto bec = std::make_unique<reco::PFBlockElementCluster>(clusterRef,reco::PFBlockElement::HGCAL);
	block.addElement(bec.get());
	simCluster2Block[simc.key()] = icp;
	simCluster2BlockIndex[simc.key()] = bec->index();
	caloParticle2SimCluster.emplace(icp,simc.key());
      }
    }
    
    auto pdgId = std::abs(CaloParticles[icp].pdgId());

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

  auto candidates = std::make_unique<reco::PFCandidateCollection>();
  // in good particle flow fashion, start from the tracks and go out
  for( unsigned itk = 0; itk < TrackCollection.size(); ++itk ) {
    auto tkRef  = TrackCollection.refAt(itk);
     // skip tracks not selected by PF
    if( PFTrackToGeneralTrack.count(itk) == 0  ) continue;
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
    
    if (useTiming_) candidate.setTime( (*trackTimeH)[tkRef], (*trackTimeErrH)[tkRef] );
    
    // bind to cluster if there is one and try to gather conversions, etc
    for( const auto& match : matches ) {      
      uint64_t hash = hashSimInfo(*(match.first));
      if( hashToSimCluster.count(hash) ) {	
	auto simcHash = hashToSimCluster[hash];
	
	if( !usedSimCluster[simcHash] ) {	 
	  if( simCluster2Block.count(simcHash) && 
	      simCluster2BlockIndex.count(simcHash) ) {
	    size_t block    = simCluster2Block.find(simcHash)->second;
	    size_t blockIdx = simCluster2BlockIndex.find(simcHash)->second;
	    edm::Ref<reco::PFBlockCollection> blockRef(blocksHandle,block);
	    candidate.addElementInBlock(blockRef,blockIdx);
	    usedSimCluster[simcHash] = true;
	  }
	}
	if( absPdgId == 11 ) { // collect brems/conv. brems
	  if( simCluster2Block.count(simcHash) ) {
	    auto block_index = simCluster2Block.find(simcHash)->second;
	    auto supercluster_index = caloParticle2SuperCluster[ block_index ];
	    if( supercluster_index != -1 ) {
	      edm::Ref<reco::PFBlockCollection> blockRef(blocksHandle,block_index);
	      for( const auto& elem : blockRef->elements() ) {
		const auto& ref = elem.clusterRef();
		if( !usedSimCluster[ref.key()] ) {
		  candidate.addElementInBlock(blockRef,elem.index());
		  usedSimCluster[ref.key()] = true;
		}
	      }
	      
              //*TODO* cluster time is not reliable at the moment, so just keep time from the track if available
              if (false) {
                const reco::PFCluster *seed = dynamic_cast<const reco::PFCluster *>((*superClustersHandle)[supercluster_index].seed().get());
                assert(seed != nullptr);
                if (seed->timeError() > 0) {
                  if (candidate.isTimeValid() && candidate.timeError() > 0) {
                    double wCand = 1.0/std::pow(candidate.timeError(),2), wSeed = 1.0/std::pow(seed->timeError(),2);
                    candidate.setTime((wCand*candidate.time() + wSeed*seed->time())/(wCand + wSeed) , 1.0f/std::sqrt(float(wCand + wSeed)));
                  } else {
                    candidate.setTime(seed->time(), seed->timeError());
                  }
                }
              }
	    }
	  }
	}
      }
    }
    usedTrack[tkRef.key()] = true;    
    // remove tracks already used by muons
    if( MuonTrackToGeneralTrack.count(itk) || absPdgId == 13)
      candidates->pop_back();
  }

  // now loop over the non-collected clusters in blocks 
  // and turn them into neutral hadrons or photons
  const auto& theblocks = *blocksHandle;
  for( unsigned ibl = 0; ibl < theblocks.size(); ++ibl ) {
    reco::PFBlockRef blref(blocksHandle,ibl);
    const auto& elements = theblocks[ibl].elements();
    for( const auto& elem : elements ) {
      const auto& ref = elem.clusterRef();
      const auto& simtruth = SimClustersTruth[ref.key()];
      reco::PFCandidate::ParticleType part_type;
      if( !usedSimCluster[ref.key()] ) {
	auto absPdgId = std::abs(simtruth.pdgId());
	switch( absPdgId ) {
	case 11:
	case 22:
	  part_type = reco::PFCandidate::gamma;
	  break;
	default:
	  part_type = reco::PFCandidate::h0;
	}
	const auto three_mom = (ref->position() - math::XYZPoint(0,0,0)).unit()*ref->energy();
	math::XYZTLorentzVector clu_p4(three_mom.x(),three_mom.y(),three_mom.z(),ref->energy());
	candidates->emplace_back(0, clu_p4, part_type);
	auto& candidate = candidates->back();
	candidate.addElementInBlock(blref,elem.index());
        candidate.setTime(ref->time(), ref->timeError());
      }
    }
  }
  
  evt.put(std::move(candidates));
}
