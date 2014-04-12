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
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

// ROOT includes
#include "TF1.h"

namespace reco { namespace modules {
    
    class HICaloCompatibleTrackSelector : public edm::EDProducer {
      
    public:
      /// constructor 
      explicit HICaloCompatibleTrackSelector(const edm::ParameterSet& cfg);
      /// destructor
      virtual ~HICaloCompatibleTrackSelector() ;
      
    private:
      typedef math::XYZPoint Point;
      typedef reco::PFCandidateCollection::const_iterator CI;
      typedef reco::TrackCollection::const_iterator TI;

      /// process one event
      void produce( edm::Event& evt, const edm::EventSetup& es ) ;
      
      void matchByDrAllowReuse(const reco::Track & trk, const edm::Handle<CaloTowerCollection> & towers, double & bestdr, double & bestpt);
      
      double matchPFCandToTrack(const edm::Handle<PFCandidateCollection> & pfCandidates, unsigned it, double trkPt);
      
      bool selectByPFCands(TI ti, const edm::Handle<TrackCollection> hSrcTrack, const edm::Handle<PFCandidateCollection> pfCandidates, bool isPFThere);
      bool selectByTowers(TI ti, const edm::Handle<TrackCollection> hSrcTrack, const edm::Handle<CaloTowerCollection> towers, bool isTowerThere);
      
      /// source collection label
      edm::InputTag srcTracks_;
      edm::InputTag srcPFCands_;
      edm::InputTag srcTower_;
      
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
      std::auto_ptr<reco::TrackCollection> selTracks_;
      std::auto_ptr<reco::TrackExtraCollection> selTrackExtras_;
      std::auto_ptr< TrackingRecHitCollection>  selHits_;
      std::auto_ptr< std::vector<Trajectory> > selTrajs_;
      std::auto_ptr< std::vector<const Trajectory *> > selTrajPtrs_;
      std::auto_ptr< TrajTrackAssociationCollection >  selTTAss_;
      reco::TrackRefProd rTracks_;
      reco::TrackExtraRefProd rTrackExtras_;
      TrackingRecHitRefProd rHits_;
      edm::RefProd< std::vector<Trajectory> > rTrajectories_;
      std::vector<reco::TrackRef> trackRefs_;

      // TF1         
      TF1 *fDeltaRTowerMatch, *fCaloComp;
      

		   };
    
  } 
}

#endif
