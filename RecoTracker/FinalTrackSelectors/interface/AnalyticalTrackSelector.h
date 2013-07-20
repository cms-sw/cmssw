#ifndef RecoAlgos_AnalyticalTrackSelector_h
#define RecoAlgos_AnalyticalTrackSelector_h
/** \class AnalyticalTrackSelector
 *
 * selects a subset of a track collection, copying extra information on demand
 * 
 * \author Paolo Azzurri, Giovanni Petrucciani 
 *
 * \version $Revision: 1.22 $
 *
 * $Id: AnalyticalTrackSelector.h,v 1.22 2013/02/28 09:57:34 muzaffar Exp $
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
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "RecoTracker/FinalTrackSelectors/src/MultiTrackSelector.h"

namespace reco { namespace modules {
    //class MultiTrackSelector;

    class AnalyticalTrackSelector : public MultiTrackSelector {
        private:
        public:
            /// constructor 
            explicit AnalyticalTrackSelector( const edm::ParameterSet & cfg ) ;
            /// destructor
            virtual ~AnalyticalTrackSelector() ;

        protected:
            typedef math::XYZPoint Point;
            /// process one event
            void produce( edm::Event& evt, const edm::EventSetup& es )  override;
 
            /// copy only the tracks, not extras and rechits (for AOD)
            bool copyExtras_;
            /// copy also trajectories and trajectory->track associations
            bool copyTrajectories_;
            /// eta restrictions
            double minEta_;
	    double maxEta_;
			
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

    };

} }

#endif
