#ifndef HICaloCompatibleTrackSelector_H
#define HICaloCompatibleTrackSelector_H

/** \class HICaloCompatibleTrackSelector
 *
 *  It selects tracks based on a cut imposed on track-calo compatibility, 
 *  and saves tracks in event. 
 * 
 *
 */

#include <utility>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

// ROOT includes
#include "TF1.h"

namespace reco {
  namespace modules {

    class HICaloCompatibleTrackSelector : public edm::stream::EDProducer<> {
    public:
      /// constructor
      explicit HICaloCompatibleTrackSelector(const edm::ParameterSet& cfg);
      /// destructor
      ~HICaloCompatibleTrackSelector() override;

    private:
      typedef math::XYZPoint Point;
      typedef reco::PFCandidateCollection::const_iterator CI;
      typedef reco::TrackCollection::const_iterator TI;

      /// process one event
      void produce(edm::Event& evt, const edm::EventSetup& es) override;

      void matchByDrAllowReuse(const reco::Track& trk,
                               const edm::Handle<CaloTowerCollection>& towers,
                               double& bestdr,
                               double& bestpt);

      double matchPFCandToTrack(const edm::Handle<PFCandidateCollection>& pfCandidates, unsigned it, double trkPt);

      bool selectByPFCands(TI ti,
                           const edm::Handle<TrackCollection> hSrcTrack,
                           const edm::Handle<PFCandidateCollection> pfCandidates,
                           bool isPFThere);
      bool selectByTowers(TI ti,
                          const edm::Handle<TrackCollection> hSrcTrack,
                          const edm::Handle<CaloTowerCollection> towers,
                          bool isTowerThere);

      /// source collection label
      edm::EDGetTokenT<reco::TrackCollection> srcTracks_;
      edm::EDGetTokenT<reco::PFCandidateCollection> srcPFCands_;
      edm::EDGetTokenT<CaloTowerCollection> srcTower_;
      edm::EDGetTokenT<std::vector<Trajectory>> srcTrackTrajs_;
      edm::EDGetTokenT<TrajTrackAssociationCollection> srcTrackTrajAssoc_;

      //
      bool applyPtDepCut_;
      bool usePFCandMatching_;
      double trkMatchPtMin_;
      double trkCompPtMin_;
      double trkEtaMax_;
      double towerPtMin_;
      double matchConeRadius_;

      bool keepAllTracks_;
      /// copy only the tracks, not extras and rechits (for AOD)
      bool copyExtras_;
      /// copy also trajectories and trajectory->track associations
      bool copyTrajectories_;

      std::string qualityToSet_;
      std::string qualityToSkip_;
      std::string qualityToMatch_;
      std::string minimumQuality_;
      bool resetQuality_;

      bool passMuons_;
      bool passElectrons_;

      // string of functional form
      std::string funcDeltaRTowerMatch_;
      std::string funcCaloComp_;

      /// storage
      std::unique_ptr<reco::TrackCollection> selTracks_;
      std::unique_ptr<reco::TrackExtraCollection> selTrackExtras_;
      std::unique_ptr<TrackingRecHitCollection> selHits_;
      std::unique_ptr<std::vector<Trajectory>> selTrajs_;
      std::unique_ptr<std::vector<const Trajectory*>> selTrajPtrs_;
      std::unique_ptr<TrajTrackAssociationCollection> selTTAss_;
      reco::TrackRefProd rTracks_;
      reco::TrackExtraRefProd rTrackExtras_;
      TrackingRecHitRefProd rHits_;
      edm::RefProd<std::vector<Trajectory>> rTrajectories_;
      std::vector<reco::TrackRef> trackRefs_;

      // TF1
      TF1 *fDeltaRTowerMatch, *fCaloComp;
    };

  }  // namespace modules
}  // namespace reco

#endif
