#ifndef RecoAlgos_CosmicTrackSelector_h
#define RecoAlgos_CosmicTrackSelector_h
/** \class CosmicTrackSelector
 *
 * selects a subset of a track collection, copying extra information on demand
 * 
 * \author Paolo Azzurri, Giovanni Petrucciani 
 *
 * \version $Revision: 1.3 $
 *
 * $Id: CosmicTrackSelector.h,v 1.3 2013/02/27 13:28:30 muzaffar Exp $
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
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"


namespace reco { namespace modules {
    
    class CosmicTrackSelector : public edm::EDProducer {
		   private:
		   public:
		     // constructor 
		     explicit CosmicTrackSelector( const edm::ParameterSet & cfg ) ;
		     // destructor
		     virtual ~CosmicTrackSelector() ;
		     
		   private:
		     typedef math::XYZPoint Point;
		     // process one event
		     void produce( edm::Event& evt, const edm::EventSetup& es ) override;
		     // return class, or -1 if rejected
		     bool select (const reco::BeamSpot &vertexBeamSpot, const reco::Track &tk);
		     // source collection label
		     edm::InputTag src_;
		     edm::InputTag beamspot_;
		     // copy only the tracks, not extras and rechits (for AOD)
		     bool copyExtras_;
		     // copy also trajectories and trajectory->track associations
		     bool copyTrajectories_;
		     
		     // save all the tracks
		     bool keepAllTracks_;
		     // do I have to set a quality bit?
		     bool setQualityBit_;
		     TrackBase::TrackQuality qualityToSet_;
		     
		     //  parameters for adapted optimal cuts on chi2 and primary vertex compatibility
		     std::vector<double> res_par_;
		     double  chi2n_par_;

		     // Impact parameter absolute cuts
		     double max_d0_;
		     double max_z0_;
		     // Trackk parameter cuts
		     double min_pt_;
		     double max_eta_;
		     // Cut on number of valid hits
		     uint32_t min_nHit_;
		     // Cut on number of valid Pixel hits
		     uint32_t min_nPixelHit_;
		     // Cuts on numbers of layers with hits/3D hits/lost hits. 
		     uint32_t min_layers_;
		     uint32_t min_3Dlayers_;
		     uint32_t max_lostLayers_;
		     
		     // storage
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
