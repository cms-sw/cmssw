#ifndef RecoAlgos_TrackMultiSelector_h
#define RecoAlgos_TrackMultiSelector_h
/** \class TrackMultiSelector
 *
 * selects a subset of a track collection, copying extra information on demand
 * 
 * \author Giovanni Petrucciani 
 *
 * \version $Revision: 1.7 $
 *
 * $Id: TrackMultiSelector.h,v 1.7 2013/02/27 13:28:30 muzaffar Exp $
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
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

namespace reco { namespace modules {

    class TrackMultiSelector : public edm::EDProducer {
        private:
            struct Block {
                std::pair<double,double> pt;
                std::pair<uint32_t,uint32_t>   vlayers, lhits;
                std::pair<double,double> chi2n;
                double d0, dz,d0Rel,dzRel;

                explicit Block(const edm::ParameterSet & cfg) ;
                private:
                template<typename T> std::pair<T,T> p2p(const edm::ParameterSet & cfg, const std::string name);
            };
        public:
            /// constructor 
            explicit TrackMultiSelector( const edm::ParameterSet & cfg ) ;
            /// destructor
            virtual ~TrackMultiSelector() ;

        private:
            typedef math::XYZPoint Point;
            /// process one event
            void produce( edm::Event& evt, const edm::EventSetup& es ) override;
            /// return class, or -1 if rejected
            short select ( const reco::Track &tk, const reco::BeamSpot &beamSpot, const std::vector<Point> &points);
            void selectVertices ( const reco::VertexCollection &vtxs, std::vector<Point> &points);
            inline bool testVtx ( const reco::Track &tk, const reco::BeamSpot  &beamSpot,
				  const std::vector<Point> &points, const Block &cut);
            /// source collection label
            edm::InputTag src_;
            edm::InputTag vertices_;
            edm::InputTag beamspot_;
            double        beamspotDZsigmas_, beamspotD0_;
            /// copy only the tracks, not extras and rechits (for AOD)
            bool copyExtras_;
            /// copy also trajectories and trajectory->track associations
            bool copyTrajectories_;
            /// split selections in more sets
            bool splitOutputs_;
            /// filter psets
            std::vector<Block> blocks_; 
            /// vertex cuts
            int32_t vtxNumber_;
            size_t  vtxTracks_;
            double  vtxChi2Prob_;
            /// output labels
            std::vector<std::string> labels_;
            /// some storage
            std::auto_ptr<reco::TrackCollection> *selTracks_;
            std::auto_ptr<reco::TrackExtraCollection> *selTrackExtras_;
            std::auto_ptr< TrackingRecHitCollection>  *selHits_;
            std::auto_ptr< std::vector<Trajectory> > *selTrajs_;
            std::auto_ptr< TrajTrackAssociationCollection >  *selTTAss_;
            std::vector<reco::TrackRefProd> rTracks_;
            std::vector<reco::TrackExtraRefProd> rTrackExtras_;
            std::vector<TrackingRecHitRefProd> rHits_;
            std::vector< edm::RefProd< std::vector<Trajectory> > > rTrajectories_;
            std::vector< std::pair<short, reco::TrackRef> > whereItWent_;

    };

} }

// template method to be implemented here?
template<typename T> std::pair<T,T> reco::modules::TrackMultiSelector::Block::p2p(const edm::ParameterSet & cfg, const std::string name) {
    typedef typename std::vector<T> Ts;
    Ts ret = cfg.getParameter<Ts>(name);
    if (ret.size() != 2) throw cms::Exception("Invalid configuration") << "Parameter '" << name << "' must be given as {min,max}";
    return std::pair<T,T>(ret[0],ret[1]);
}
#endif
