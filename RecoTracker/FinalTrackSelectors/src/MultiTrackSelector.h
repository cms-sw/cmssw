#ifndef RecoAlgos_MultiTrackSelector_h
#define RecoAlgos_MultiTrackSelector_h
/** \class MultiTrackSelector
 *
 * selects a subset of a track collection, copying extra information on demand
 * 
 * \author David Lange
 *
 * \version $Revision: 1.1.2.1 $
 *
 * $Id: MultiTrackSelector.h,v 1.1.2.1 2011/06/18 09:13:46 gpetrucc Exp $
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

namespace reco { namespace modules {

    class MultiTrackSelector : public edm::EDProducer {
        private:
        public:
            /// constructor 
            explicit MultiTrackSelector( const edm::ParameterSet & cfg ) ;
            /// destructor
            virtual ~MultiTrackSelector() ;

        private:
            typedef math::XYZPoint Point;
            /// process one event
            void produce( edm::Event& evt, const edm::EventSetup& es ) ;
            /// return class, or -1 if rejected
            bool select (unsigned tsNum,
			 const reco::BeamSpot &vertexBeamSpot, 
			 const reco::Track &tk, 
			 const std::vector<Point> &points);
            void selectVertices ( unsigned int tsNum,
				  const reco::VertexCollection &vtxs, 
				  std::vector<Point> &points);
            /// source collection label
            edm::InputTag src_;
            edm::InputTag beamspot_;
            bool          useVertices_;
            edm::InputTag vertices_;
            
            /// do I have to set a quality bit?
	    std::vector<bool> setQualityBit_;
	    std::vector<TrackBase::TrackQuality> qualityToSet_;

            /// vertex cuts
	    std::vector<int32_t> vtxNumber_;
	    std::vector<StringCutObjectSelector<reco::Vertex> > vertexCut_;

	    //  parameters for adapted optimal cuts on chi2 and primary vertex compatibility
	    std::vector< std::vector<double> > res_par_;
	    std::vector< double > chi2n_par_;
	    std::vector< std::vector<double> > d0_par1_;
	    std::vector< std::vector<double> > dz_par1_;
	    std::vector< std::vector<double> > d0_par2_;
	    std::vector< std::vector<double> > dz_par2_;
	    // Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
            std::vector<bool> applyAdaptedPVCuts_;
			
            /// Impact parameter absolute cuts
            std::vector<double> max_d0_;
            std::vector<double> max_z0_;
            std::vector<double> nSigmaZ_;

            /// Cuts on numbers of layers with hits/3D hits/lost hits. 
	    std::vector<uint32_t> min_layers_;
	    std::vector<uint32_t> min_3Dlayers_;
	    std::vector<uint32_t> max_lostLayers_;
	    
	    // Flag and absolute cuts if no PV passes the selection
	    std::vector<double> max_d0NoPV_;
	    std::vector<double> max_z0NoPV_;
	    std::vector<bool> applyAbsCutsIfNoPV_;
	    //if true, selector flags but does not select 
	    std::vector<bool> keepAllTracks_;

	    // allow one of the previous psets to be used as a prefilter
	    std::vector<unsigned int> preFilter_;
	    std::vector<std::string> name_;
    };

} }

#endif
