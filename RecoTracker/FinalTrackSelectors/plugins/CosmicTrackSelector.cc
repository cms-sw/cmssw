/** \class CosmicTrackSelector
 *
 * selects a subset of a track collection, copying extra information on demand
 * 
 * \author Paolo Azzurri, Giovanni Petrucciani 
 *
 *
 *
 */

#include <utility>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include "FWCore/Framework/interface/stream/EDProducer.h"
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

using namespace reco;

    class dso_hidden CosmicTrackSelector final : public edm::stream::EDProducer<> {
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
                     edm::EDGetTokenT<reco::TrackCollection> src_;
                     edm::EDGetTokenT<reco::BeamSpot> beamspot_;
		     // copy only the tracks, not extras and rechits (for AOD)
		     bool copyExtras_;
		     // copy also trajectories and trajectory->track associations
		     bool copyTrajectories_;
                     edm::EDGetTokenT<std::vector<Trajectory> >        srcTraj_;
                     edm::EDGetTokenT<TrajTrackAssociationCollection > srcTass_;
		     
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
    
#include <Math/DistFunc.h>
#include "TMath.h"

CosmicTrackSelector::CosmicTrackSelector( const edm::ParameterSet & cfg ) :
  src_( consumes<reco::TrackCollection>( cfg.getParameter<edm::InputTag>( "src" ) ) ),
  beamspot_( consumes<reco::BeamSpot>( cfg.getParameter<edm::InputTag>( "beamspot" ) ) ),
  copyExtras_(cfg.getUntrackedParameter<bool>("copyExtras", false)),
  copyTrajectories_(cfg.getUntrackedParameter<bool>("copyTrajectories", false)),
  keepAllTracks_( cfg.exists("keepAllTracks") ?
		  cfg.getParameter<bool>("keepAllTracks") :
		  false ),  // as this is what you expect from a well behaved selector
  setQualityBit_( false ),
  qualityToSet_( TrackBase::undefQuality ),
  chi2n_par_( cfg.getParameter<double>("chi2n_par") ),
  // Impact parameter absolute cuts.
  max_d0_(cfg.getParameter<double>("max_d0")),
  max_z0_(cfg.getParameter<double>("max_z0")),
  // Track parameter cuts.
  min_pt_(cfg.getParameter<double>("min_pt")),
  max_eta_(cfg.getParameter<double>("max_eta")),
  // Cut on number of valid hits
  min_nHit_(cfg.getParameter<uint32_t>("min_nHit")),
  // Cut on number of valid hits
  min_nPixelHit_(cfg.getParameter<uint32_t>("min_nPixelHit")),
  // Cuts on numbers of layers with hits/3D hits/lost hits.
  min_layers_(cfg.getParameter<uint32_t>("minNumberLayers") ),
  min_3Dlayers_(cfg.getParameter<uint32_t>("minNumber3DLayers") ),
  max_lostLayers_(cfg.getParameter<uint32_t>("maxNumberLostLayers") )
{
  if (cfg.exists("qualityBit")) {
    std::string qualityStr = cfg.getParameter<std::string>("qualityBit");
    if (qualityStr != "") {
      setQualityBit_ = true;
      qualityToSet_  = TrackBase::qualityByName(cfg.getParameter<std::string>("qualityBit"));
    }
  }
  if (keepAllTracks_ && !setQualityBit_) throw cms::Exception("Configuration") << 
    "If you set 'keepAllTracks' to true, you must specify which qualityBit to set.\n";
  if (setQualityBit_ && (qualityToSet_ == TrackBase::undefQuality)) throw cms::Exception("Configuration") <<
    "You can't set the quality bit " << cfg.getParameter<std::string>("qualityBit") << " as it is 'undefQuality' or unknown.\n";
  
  std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
  produces<reco::TrackCollection>().setBranchAlias( alias + "Tracks");
  if (copyExtras_) {
    produces<reco::TrackExtraCollection>().setBranchAlias( alias + "TrackExtras");
    produces<TrackingRecHitCollection>().setBranchAlias( alias + "RecHits");
  }
  if (copyTrajectories_) {
    srcTraj_ = consumes<std::vector<Trajectory> >(cfg.getParameter<edm::InputTag>( "src" ));
    srcTass_ = consumes<TrajTrackAssociationCollection>(cfg.getParameter<edm::InputTag>( "src" ));
    produces< std::vector<Trajectory> >().setBranchAlias( alias + "Trajectories");
    produces< TrajTrackAssociationCollection >().setBranchAlias( alias + "TrajectoryTrackAssociations");
  }
  
}

CosmicTrackSelector::~CosmicTrackSelector() {
}

void CosmicTrackSelector::produce( edm::Event& evt, const edm::EventSetup& es ) 
{
  using namespace std; 
  using namespace edm;
  using namespace reco;
  
  Handle<TrackCollection> hSrcTrack;
  Handle< vector<Trajectory> > hTraj;
  Handle< vector<Trajectory> > hTrajP;
  Handle< TrajTrackAssociationCollection > hTTAss;
  
  // looking for the beam spot
  edm::Handle<reco::BeamSpot> hBsp;
  evt.getByToken(beamspot_, hBsp);
  reco::BeamSpot vertexBeamSpot;
  vertexBeamSpot = *hBsp;
  
  // Get tracks 
  evt.getByToken( src_, hSrcTrack );
  
  selTracks_ = auto_ptr<TrackCollection>(new TrackCollection());
  rTracks_ = evt.getRefBeforePut<TrackCollection>();      
  if (copyExtras_) {
    selTrackExtras_ = auto_ptr<TrackExtraCollection>(new TrackExtraCollection());
    selHits_ = auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
    rHits_ = evt.getRefBeforePut<TrackingRecHitCollection>();
    rTrackExtras_ = evt.getRefBeforePut<TrackExtraCollection>();
  }
  
  if (copyTrajectories_) trackRefs_.resize(hSrcTrack->size());
  
  // Loop over tracks
  size_t current = 0;
  for (TrackCollection::const_iterator it = hSrcTrack->begin(), ed = hSrcTrack->end(); it != ed; ++it, ++current) {
    const Track & trk = * it;
    // Check if this track passes cuts
    bool ok = select(vertexBeamSpot, trk);
    if (!ok) {
      if (copyTrajectories_) trackRefs_[current] = reco::TrackRef();
      if (!keepAllTracks_) continue;
    }
    selTracks_->push_back( Track( trk ) ); // clone and store
    if (ok && setQualityBit_) selTracks_->back().setQuality(qualityToSet_);
    if (copyExtras_) {
      // TrackExtras
      selTrackExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
					      trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
					      trk.outerStateCovariance(), trk.outerDetId(),
					      trk.innerStateCovariance(), trk.innerDetId(),
					      trk.seedDirection(), trk.seedRef() ) );
      selTracks_->back().setExtra( TrackExtraRef( rTrackExtras_, selTrackExtras_->size() - 1) );
      TrackExtra & tx = selTrackExtras_->back();
      tx.setResiduals(trk.residuals());
      // TrackingRecHits
      auto const firstHitIndex = selHits_->size();
      for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
	selHits_->push_back( (*hit)->clone() );
      }
      tx.setHits( rHits_, firstHitIndex, selHits_->size() - firstHitIndex );
    }
    if (copyTrajectories_) {
      trackRefs_[current] = TrackRef(rTracks_, selTracks_->size() - 1);
    }
    }
  if ( copyTrajectories_ ) {
    Handle< vector<Trajectory> > hTraj;
    Handle< TrajTrackAssociationCollection > hTTAss;
    evt.getByToken(srcTass_, hTTAss);
    evt.getByToken(srcTraj_, hTraj);
    selTrajs_ = auto_ptr< vector<Trajectory> >(new vector<Trajectory>()); 
    rTrajectories_ = evt.getRefBeforePut< vector<Trajectory> >();
    selTTAss_ = auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection(&evt.productGetter()));
    for (size_t i = 0, n = hTraj->size(); i < n; ++i) {
      Ref< vector<Trajectory> > trajRef(hTraj, i);
      TrajTrackAssociationCollection::const_iterator match = hTTAss->find(trajRef);
      if (match != hTTAss->end()) {
	const Ref<TrackCollection> &trkRef = match->val; 
	short oldKey = static_cast<short>(trkRef.key());
	if (trackRefs_[oldKey].isNonnull()) {
	  selTrajs_->push_back( Trajectory(*trajRef) );
	  selTTAss_->insert ( Ref< vector<Trajectory> >(rTrajectories_, selTrajs_->size() - 1), trackRefs_[oldKey] );
	}
      }
    }
  }
  
  static const std::string emptyString;
  evt.put(selTracks_);
  if (copyExtras_ ) {
    evt.put(selTrackExtras_); 
    evt.put(selHits_);
  }
  if ( copyTrajectories_ ) {
    evt.put(selTrajs_);
    evt.put(selTTAss_);
  }
}


bool CosmicTrackSelector::select(const reco::BeamSpot &vertexBeamSpot, const reco::Track &tk) {
  // Decide if the given track passes selection cuts.
  
  using namespace std; 
  
  // Cuts on numbers of layers with hits/3D hits/lost hits.
  uint32_t nlayers     = tk.hitPattern().trackerLayersWithMeasurement();
  uint32_t nlayers3D   = tk.hitPattern().pixelLayersWithMeasurement() +
      tk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
  uint32_t nlayersLost = tk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
  
  // Get the number of valid hits and PixelHits
  uint32_t nHit = 0;
  uint32_t nPixelHit = 0;
  for ( trackingRecHit_iterator recHit = tk.recHitsBegin(); recHit != tk.recHitsEnd(); ++recHit ) {
    if ( !((*recHit)->isValid()) ) continue; 
    ++nHit;
    DetId id((*recHit)->geographicalId());
    if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelBarrel 
	 || (unsigned int)id.subdetId() == PixelSubdetector::PixelEndcap )
      ++nPixelHit;
  }  
  
  // Cut on the number of valid hits
  if (nHit < min_nHit_) return false;
  // Cut on the number of valid Pixel hits
  if (nPixelHit < min_nPixelHit_) return false;
  if (nlayers < min_layers_) return false;
  if (nlayers3D < min_3Dlayers_) return false;
  if (nlayersLost > max_lostLayers_) return false;
  
  // Get track parameters
  double pt = tk.pt(),eta = tk.eta(), chi2n =  tk.normalizedChi2();
  double d0 = -tk.dxy(vertexBeamSpot.position()), dz = tk.dz();
  
  // Absolute cuts on all tracks impact parameters with respect to beam-spot.
  if (abs(d0) > max_d0_) return false;
  if (abs(dz) > max_z0_) return false;
  
  // optimized cuts adapted to the track eta, pt and  chiquare/ndof 
  if (abs(eta) > max_eta_) return false;
  if (pt < min_pt_) return false;
  if (chi2n > chi2n_par_*nlayers) return false;

  
  else    
    return true;
  
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CosmicTrackSelector);

