#ifndef RecoAlgos_AnalyticalTrackSelector_h
#define RecoAlgos_AnalyticalTrackSelector_h
/** \class AnalyticalTrackSelector
 *
 * selects a subset of a track collection, copying extra information on demand
 * 
 * \author Paolo Azzurri, Giovanni Petrucciani 
 *
 * \version $Revision: 1.6 $
 *
 * $Id: AnalyticalTrackSelector.h,v 1.6 2008/03/11 00:40:53 gpetrucc Exp $
 *
 */

#include <utility>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"


namespace reco { namespace modules {

    class AnalyticalTrackSelector : public edm::EDProducer {
        private:
        public:
            /// constructor 
            explicit AnalyticalTrackSelector( const edm::ParameterSet & cfg ) ;
            /// destructor
            virtual ~AnalyticalTrackSelector() ;

        private:
            typedef math::XYZPoint Point;
            /// process one event
            void produce( edm::Event& evt, const edm::EventSetup& es ) ;
            /// return class, or -1 if rejected
            bool select (const reco::BeamSpot &vertexBeamSpot, const reco::Track &tk, const std::vector<Point> &points);
            void selectVertices ( const reco::VertexCollection &vtxs, std::vector<Point> &points);
            /// source collection label
            edm::InputTag src_;
            edm::InputTag beamspot_;
            edm::InputTag vertices_;
            /// copy only the tracks, not extras and rechits (for AOD)
            bool copyExtras_;
            /// copy also trajectories and trajectory->track associations
            bool copyTrajectories_;

            /// save all the tracks
            bool keepAllTracks_;
            /// do I have to set a quality bit?
            bool setQualityBit_;
            TrackBase::TrackQuality qualityToSet_;

            /// vertex cuts
            int32_t vtxNumber_;
            size_t  vtxTracks_;
            double  vtxChi2Prob_;
            /// paremeters for track cuts
			std::vector<double> res_par_;
            double  chi2n_par_;
			std::vector<double> d0_par1_;
			std::vector<double> dz_par1_;
			std::vector<double> d0_par2_;
			std::vector<double> dz_par2_;

	    /// Minimum number of hits
			uint32_t min_layers_;

            /// storage
            std::auto_ptr<reco::TrackCollection> selTracks_;
            std::auto_ptr<reco::TrackExtraCollection> selTrackExtras_;
            std::auto_ptr< TrackingRecHitCollection>  selHits_;
            std::auto_ptr< std::vector<Trajectory> > selTrajs_;
            std::auto_ptr< TrajTrackAssociationCollection >  selTTAss_;
            reco::TrackRefProd rTracks_;
            reco::TrackExtraRefProd rTrackExtras_;
            TrackingRecHitRefProd rHits_;
            edm::RefProd< std::vector<Trajectory> > rTrajectories_;
            std::vector<reco::TrackRef> trackRefs_;

    };

} }

#endif
