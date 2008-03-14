#include "RecoTracker/FinalTrackSelectors/interface/AnalyticalTrackSelector.h"

#include <Math/DistFunc.h>

using reco::modules::AnalyticalTrackSelector;

AnalyticalTrackSelector::AnalyticalTrackSelector( const edm::ParameterSet & cfg ) :
    src_( cfg.getParameter<edm::InputTag>( "src" ) ),
    beamspot_( cfg.getParameter<edm::InputTag>( "beamspot" ) ),
    vertices_( cfg.getParameter<edm::InputTag>( "vertices" ) ),
    copyExtras_(cfg.getUntrackedParameter<bool>("copyExtras", false)),
    copyTrajectories_(cfg.getUntrackedParameter<bool>("copyTrajectories", false)),
    vtxNumber_( cfg.getParameter<int32_t>("vtxNumber") ),
    vtxTracks_( cfg.getParameter<uint32_t>("vtxTracks") ),
    vtxChi2Prob_( cfg.getParameter<double>("vtxChi2Prob") ),
    chi2n_par_( cfg.getParameter<double>("chi2n_par") ),
	d0_par1_(cfg.getParameter< std::vector<double> >("d0_par1")),
	dz_par1_(cfg.getParameter< std::vector<double> >("dz_par1")),
	d0_par2_(cfg.getParameter< std::vector<double> >("d0_par2")),
	dz_par2_(cfg.getParameter< std::vector<double> >("dz_par2"))
{
 
    std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
	produces<reco::TrackCollection>().setBranchAlias( alias + "Tracks");
	if (copyExtras_) {
	  produces<reco::TrackExtraCollection>().setBranchAlias( alias + "TrackExtras");
	  produces<TrackingRecHitCollection>().setBranchAlias( alias + "RecHits");
	  if (copyTrajectories_) {
		produces< std::vector<Trajectory> >().setBranchAlias( alias + "Trajectories");
		produces< TrajTrackAssociationCollection >().setBranchAlias( alias + "TrajectoryTrackAssociations");
	  }
	}
	selTracks_ = std::auto_ptr<reco::TrackCollection>(new reco::TrackCollection());
	selTrackExtras_ = std::auto_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection());
	selHits_ = std::auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
	selTrajs_ = std::auto_ptr< std::vector<Trajectory> >(new std::vector<Trajectory>()); 
	selTTAss_ = std::auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());
 
}

AnalyticalTrackSelector::~AnalyticalTrackSelector() {
}

void AnalyticalTrackSelector::produce( edm::Event& evt, const edm::EventSetup& es ) 
{
    using namespace std; 
    using namespace edm;
    using namespace reco;

    Handle<TrackCollection> hSrcTrack;
    Handle< vector<Trajectory> > hTraj;
    Handle< TrajTrackAssociationCollection > hTTAss;

	// looking for the beam spot
	edm::Handle<reco::BeamSpot> hBsp;
    evt.getByLabel(beamspot_, hBsp);
	reco::BeamSpot vertexBeamSpot;
	vertexBeamSpot = *hBsp;
	
    edm::Handle<reco::VertexCollection> hVtx;
    evt.getByLabel(vertices_, hVtx);
    std::vector<Point> points;
    selectVertices(*hVtx, points);

    evt.getByLabel( src_, hSrcTrack );

	selTracks_ = auto_ptr<TrackCollection>(new TrackCollection());
	rTracks_ = evt.getRefBeforePut<TrackCollection>();      
	if (copyExtras_) {
	  selTrackExtras_ = auto_ptr<TrackExtraCollection>(new TrackExtraCollection());
	  selHits_ = auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
	  rHits_ = evt.getRefBeforePut<TrackingRecHitCollection>();
	  rTrackExtras_ = evt.getRefBeforePut<TrackExtraCollection>();
	}

    if (copyTrajectories_) trackRefs_.resize(hSrcTrack->size());
    size_t current = 0;
    for (TrackCollection::const_iterator it = hSrcTrack->begin(), ed = hSrcTrack->end(); it != ed; ++it, ++current) {
        const Track & trk = * it;
        bool ok = select(vertexBeamSpot, trk, points); 
        if (!ok) {
            if (copyTrajectories_) trackRefs_[current] = reco::TrackRef();
            continue;
        }
		selTracks_->push_back( Track( trk ) ); // clone and store
        if (!copyExtras_) continue;

        // TrackExtras
        selTrackExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
                    trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
                    trk.outerStateCovariance(), trk.outerDetId(),
                    trk.innerStateCovariance(), trk.innerDetId(),
                    trk.seedDirection() ) );
        selTracks_->back().setExtra( TrackExtraRef( rTrackExtras_, selTrackExtras_->size() - 1) );
        TrackExtra & tx = selTrackExtras_->back();
        // TrackingRecHits
        for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
            selHits_->push_back( (*hit)->clone() );
            tx.add( TrackingRecHitRef( rHits_, selHits_->size() - 1) );
        }
        if (copyTrajectories_) {
            trackRefs_[current] = TrackRef(rTracks_, selTracks_->size() - 1);
        }
    }
    if ( copyTrajectories_ ) {
        Handle< vector<Trajectory> > hTraj;
        Handle< TrajTrackAssociationCollection > hTTAss;
        evt.getByLabel(src_, hTTAss);
        evt.getByLabel(src_, hTraj);
		selTrajs_ = auto_ptr< vector<Trajectory> >(new vector<Trajectory>()); 
		rTrajectories_ = evt.getRefBeforePut< vector<Trajectory> >();
		selTTAss_ = auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());
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
	if (copyExtras_ ) {
	  if ( copyTrajectories_ ) {
		evt.put(selTrajs_);
		evt.put(selTTAss_);
	  }
	}
}


bool AnalyticalTrackSelector::select(const reco::BeamSpot &vertexBeamSpot, const reco::Track &tk, const std::vector<Point> &points) {
   using namespace std; 
   uint32_t nhits = tk.numberOfValidHits();
   double pt = tk.pt(),eta = tk.eta(), chi2n =  tk.normalizedChi2();
   double d0 = -tk.dxy(vertexBeamSpot.position()), d0E =  tk.d0Error(),dz = tk.dz(), dzE =  tk.dzError();
   // nominal d0 resolution for the track pt
   double nomd0E = sqrt(0.003*0.003+(0.01/max(pt,1e-9))*(0.01/max(pt,1e-9)));
   // nominal z0 resolution for the track pt and eta
   double nomdzE = nomd0E*(std::cosh(eta));
   //cut on chiquare/ndof && on d0 compatibility with beam line
   if (chi2n <= chi2n_par_*nhits &&
	   abs(d0) < pow(d0_par1_[0]*nhits,d0_par1_[1])*nomd0E &&
	   abs(d0) < pow(d0_par2_[0]*nhits,d0_par2_[1])*d0E ) {
	 //no vertex, wide z cuts
	 if (points.empty()) { 
	   if ( abs(dz) < 15.9 ) return true;
	 }
	 // z compatibility with a vertex
	 for (std::vector<Point>::const_iterator point = points.begin(), end = points.end(); point != end; ++point) {
	   if (
		   abs(dz-(point->z())) < pow(dz_par1_[0]*nhits,dz_par1_[1])*nomdzE &&
		   abs(dz-(point->z())) < pow(dz_par2_[0]*nhits,dz_par2_[1])*dzE ) return true;
	 }
   }
   return false;
}
void AnalyticalTrackSelector::selectVertices(const reco::VertexCollection &vtxs, std::vector<Point> &points) {
    using namespace reco;
    using ROOT::Math::chisquared_prob;
    int32_t toTake = vtxNumber_; 
    for (VertexCollection::const_iterator it = vtxs.begin(), ed = vtxs.end(); it != ed; ++it) {
        if ((it->tracksSize() >= vtxTracks_)  && 
                ( (it->chi2() == 0.0) || (chisquared_prob(it->chi2(), it->ndof()) >= vtxChi2Prob_) ) ) {
            points.push_back(it->position()); 
            toTake--; if (toTake == 0) break;
        }
    }
}
