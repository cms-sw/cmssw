#include "RecoTracker/FinalTrackSelectors/interface/AnalyticalTrackSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <Math/DistFunc.h>
#include "TMath.h"

using reco::modules::AnalyticalTrackSelector;

AnalyticalTrackSelector::AnalyticalTrackSelector( const edm::ParameterSet & cfg ) :
  src_( cfg.getParameter<edm::InputTag>( "src" ) ),
  beamspot_( cfg.getParameter<edm::InputTag>( "beamspot" ) ),
  vertices_( cfg.getParameter<edm::InputTag>( "vertices" ) ),
  copyExtras_(cfg.getUntrackedParameter<bool>("copyExtras", false)),
  copyTrajectories_(cfg.getUntrackedParameter<bool>("copyTrajectories", false)),
  keepAllTracks_( cfg.exists("keepAllTracks") ?
		  cfg.getParameter<bool>("keepAllTracks") :
		  false ),  // as this is what you expect from a well behaved selector
  setQualityBit_( false ),
  qualityToSet_( TrackBase::undefQuality ),
  // parameters for vertex selection
  vtxNumber_( cfg.getParameter<int32_t>("vtxNumber") ),
  vertexCut_(cfg.getParameter<std::string>("vertexCut")),
  //  parameters for adapted optimal cuts on chi2 and primary vertex compatibility
  res_par_(cfg.getParameter< std::vector<double> >("res_par") ),
  chi2n_par_( cfg.getParameter<double>("chi2n_par") ),
  d0_par1_(cfg.getParameter< std::vector<double> >("d0_par1")),
  dz_par1_(cfg.getParameter< std::vector<double> >("dz_par1")),
  d0_par2_(cfg.getParameter< std::vector<double> >("d0_par2")),
  dz_par2_(cfg.getParameter< std::vector<double> >("dz_par2")),
  // Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
  applyAdaptedPVCuts_(cfg.getParameter<bool>("applyAdaptedPVCuts")),
  // Impact parameter absolute cuts.
  max_d0_(cfg.getParameter<double>("max_d0")),
  max_z0_(cfg.getParameter<double>("max_z0")),
  nSigmaZ_(cfg.getParameter<double>("nSigmaZ")),
  // Cuts on numbers of layers with hits/3D hits/lost hits.
  min_layers_(cfg.getParameter<uint32_t>("minNumberLayers") ),
  min_3Dlayers_(cfg.getParameter<uint32_t>("minNumber3DLayers") ),
  max_lostLayers_(cfg.getParameter<uint32_t>("maxNumberLostLayers")),
  // Flag to apply absolute cuts if no PV passes the selection
  applyAbsCutsIfNoPV_(cfg.getParameter<bool>("applyAbsCutsIfNoPV"))
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
  if (applyAbsCutsIfNoPV_) {
    max_d0NoPV_ = cfg.getParameter<double>("max_d0NoPV");
    max_z0NoPV_ = cfg.getParameter<double>("max_z0NoPV");
  }

  std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
  produces<reco::TrackCollection>().setBranchAlias( alias + "Tracks");
  if (copyExtras_) {
    produces<reco::TrackExtraCollection>().setBranchAlias( alias + "TrackExtras");
    produces<TrackingRecHitCollection>().setBranchAlias( alias + "RecHits");
  }
  if (copyTrajectories_) {
    produces< std::vector<Trajectory> >().setBranchAlias( alias + "Trajectories");
    produces< TrajTrackAssociationCollection >().setBranchAlias( alias + "TrajectoryTrackAssociations");
  }
 
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
  Handle< vector<Trajectory> > hTrajP;
  Handle< TrajTrackAssociationCollection > hTTAss;

  // looking for the beam spot
  edm::Handle<reco::BeamSpot> hBsp;
  evt.getByLabel(beamspot_, hBsp);
  reco::BeamSpot vertexBeamSpot;
  vertexBeamSpot = *hBsp;
	
  // Select good primary vertices for use in subsequent track selection
  edm::Handle<reco::VertexCollection> hVtx;
  evt.getByLabel(vertices_, hVtx);
  std::vector<Point> points;
  selectVertices(*hVtx, points);
  // Debug 
  LogDebug("SelectVertex") << points.size() << " good pixel vertices";

  // Get tracks 
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

  // Loop over tracks
  size_t current = 0;
  for (TrackCollection::const_iterator it = hSrcTrack->begin(), ed = hSrcTrack->end(); it != ed; ++it, ++current) {
    const Track & trk = * it;
    // Check if this track passes cuts

    LogTrace("TrackSelection") << "ready to check track with pt="<< trk.pt() ;

    bool ok = select(vertexBeamSpot, trk, points);
    if (!ok) {

      LogTrace("TrackSelection") << "track with pt="<< trk.pt() << " NOT selected";

      if (copyTrajectories_) trackRefs_[current] = reco::TrackRef();
      if (!keepAllTracks_) continue;
    }
    LogTrace("TrackSelection") << "track with pt="<< trk.pt() << " selected";
    selTracks_->push_back( Track( trk ) ); // clone and store
    if (ok && setQualityBit_) {
      selTracks_->back().setQuality(qualityToSet_);
      if (!points.empty()) {
	if (qualityToSet_==TrackBase::loose) selTracks_->back().setQuality(TrackBase::looseSetWithPV);
	else if (qualityToSet_==TrackBase::highPurity) selTracks_->back().setQuality(TrackBase::highPuritySetWithPV);
      }
    }
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
      for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
	selHits_->push_back( (*hit)->clone() );
	tx.add( TrackingRecHitRef( rHits_, selHits_->size() - 1) );
      }
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
  if ( copyTrajectories_ ) {
    evt.put(selTrajs_);
    evt.put(selTTAss_);
  }
}


bool AnalyticalTrackSelector::select(const reco::BeamSpot &vertexBeamSpot, const reco::Track &tk, const std::vector<Point> &points) {
  // Decide if the given track passes selection cuts.

  using namespace std; 

  if ( tk.ndof() < 1E-5 ) return false;

  // Cuts on numbers of layers with hits/3D hits/lost hits.
  uint32_t nlayers     = tk.hitPattern().trackerLayersWithMeasurement();
  uint32_t nlayers3D   = tk.hitPattern().pixelLayersWithMeasurement() +
    tk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
  uint32_t nlayersLost = tk.hitPattern().trackerLayersWithoutMeasurement();
  LogDebug("TrackSelection") << "cuts on nlayers: " << nlayers << " " << nlayers3D << " " << nlayersLost << " vs " 
			     << min_layers_ << " " << min_3Dlayers_ << " " << max_lostLayers_;
  if (nlayers < min_layers_) return false;
  if (nlayers3D < min_3Dlayers_) return false;
  if (nlayersLost > max_lostLayers_) return false;
  LogTrace("TrackSelection") << "cuts on nlayers passed";

  double chi2n =  tk.normalizedChi2();

  int count1dhits = 0;
  for (trackingRecHit_iterator ith = tk.recHitsBegin(), edh = tk.recHitsEnd(); ith != edh; ++ith) {
    const TrackingRecHit * hit = ith->get();
    DetId detid = hit->geographicalId();
    if (hit->isValid()) {
      if (typeid(*hit) == typeid(SiStripRecHit1D)) ++count1dhits;
    }
  }
  if (count1dhits > 0) {
    double chi2 = tk.chi2();
    double ndof = tk.ndof();
    chi2n = (chi2+count1dhits)/double(ndof+count1dhits);
  }
  // For each 1D rechit, the chi^2 and ndof is increased by one.  This is a way of retaining approximately
  // the same normalized chi^2 distribution as with 2D rechits.
  if (chi2n > chi2n_par_*nlayers) return false;

  // Get track parameters
  double pt = tk.pt(), eta = tk.eta();
  double d0 = -tk.dxy(vertexBeamSpot.position()), d0E =  tk.d0Error(),
    dz = tk.dz(vertexBeamSpot.position()), dzE =  tk.dzError();

  // parametrized d0 resolution for the track pt
  double nomd0E = sqrt(res_par_[0]*res_par_[0]+(res_par_[1]/max(pt,1e-9))*(res_par_[1]/max(pt,1e-9)));
  // parametrized z0 resolution for the track pt and eta
  double nomdzE = nomd0E*(std::cosh(eta));

  double dzCut = min( pow(dz_par1_[0]*nlayers,dz_par1_[1])*nomdzE, 
		      pow(dz_par2_[0]*nlayers,dz_par2_[1])*dzE );
  double d0Cut = min( pow(d0_par1_[0]*nlayers,d0_par1_[1])*nomd0E, 
		      pow(d0_par2_[0]*nlayers,d0_par2_[1])*d0E );


  // ---- PrimaryVertex compatibility cut
  bool primaryVertexZCompatibility(false);   
  bool primaryVertexD0Compatibility(false);   

  if (points.empty()) { //If not primaryVertices are reconstructed, check just the compatibility with the BS
    //z0 within (n sigma + dzCut) of the beam spot z, if no good vertex is found
    if ( abs(dz) < hypot(vertexBeamSpot.sigmaZ()*nSigmaZ_,dzCut) ) primaryVertexZCompatibility = true;  
    // d0 compatibility with beam line
    if (abs(d0) < d0Cut) primaryVertexD0Compatibility = true;     
  }

  for (std::vector<Point>::const_iterator point = points.begin(), end = points.end(); point != end; ++point) {
    LogTrace("TrackSelection") << "Test track w.r.t. vertex with z position " << point->z();
    if(primaryVertexZCompatibility && primaryVertexD0Compatibility) break;
    double dzPV = tk.dz(*point); //re-evaluate the dz with respect to the vertex position
    double d0PV = tk.dxy(*point); //re-evaluate the dxy with respect to the vertex position
    if (abs(dzPV) < dzCut)  primaryVertexZCompatibility = true;
    if (abs(d0PV) < d0Cut) primaryVertexD0Compatibility = true;     
    LogTrace("TrackSelection") << "distances " << dzPV << " " << d0PV << " vs " << dzCut << " " << d0Cut;
  }

  if (points.empty() && applyAbsCutsIfNoPV_) {
    if ( abs(dz) > max_z0NoPV_ || abs(d0) > max_d0NoPV_) return false;
  }  else {
    // Absolute cuts on all tracks impact parameters with respect to beam-spot.
    // If BS is not compatible, verify if at least the reco-vertex is compatible (useful for incorrect BS settings)
    if (abs(d0) > max_d0_ && !primaryVertexD0Compatibility) return false;
    LogTrace("TrackSelection") << "absolute cuts on d0 passed";
    if (abs(dz) > max_z0_ && !primaryVertexZCompatibility) return false;
    LogTrace("TrackSelection") << "absolute cuts on dz passed";
  }

  LogTrace("TrackSelection") << "cuts on PV: apply adapted PV cuts? " << applyAdaptedPVCuts_ 
			     << " d0 compatibility? " << primaryVertexD0Compatibility  
			     << " z compatibility? " << primaryVertexZCompatibility ;

  if (applyAdaptedPVCuts_) {
    return (primaryVertexD0Compatibility && primaryVertexZCompatibility);
  } else {
    return true;     
  }

}

void AnalyticalTrackSelector::selectVertices(const reco::VertexCollection &vtxs, std::vector<Point> &points) {
  // Select good primary vertices
  using namespace reco;
  int32_t toTake = vtxNumber_; 
  for (VertexCollection::const_iterator it = vtxs.begin(), ed = vtxs.end(); it != ed; ++it) {

    LogDebug("SelectVertex") << " select vertex with z position " << it->z() << " " 
			     << it->chi2() << " " << it->ndof() << " " << TMath::Prob(it->chi2(), static_cast<int32_t>(it->ndof()));
    Vertex vtx = *it;
    bool pass = vertexCut_( vtx );
    if( pass ) { 
      points.push_back(it->position()); 
      LogTrace("SelectVertex") << " SELECTED vertex with z position " << it->z();
      toTake--; if (toTake == 0) break;
    }
  }
}
