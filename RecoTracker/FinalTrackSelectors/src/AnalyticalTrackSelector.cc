#include "RecoTracker/FinalTrackSelectors/interface/AnalyticalTrackSelector.h"
#include "RecoTracker/FinalTrackSelectors/src/MultiTrackSelector.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <Math/DistFunc.h>
#include "TMath.h"

using reco::modules::AnalyticalTrackSelector;

AnalyticalTrackSelector::AnalyticalTrackSelector( const edm::ParameterSet & cfg ) : MultiTrackSelector( )
{
    //Spoof the pset for each track selector!
    //Size is always 1!!!
    qualityToSet_.reserve(1);
    vtxNumber_.reserve(1);
    vertexCut_.reserve(1);
    res_par_.reserve(1);
    chi2n_par_.reserve(1);
    chi2n_no1Dmod_par_.reserve(1);
    d0_par1_.reserve(1);
    dz_par1_.reserve(1);
    d0_par2_.reserve(1);
    dz_par2_.reserve(1);
    applyAdaptedPVCuts_.reserve(1);
    max_d0_.reserve(1);
    max_z0_.reserve(1);
    nSigmaZ_.reserve(1);
    min_layers_.reserve(1);
    min_3Dlayers_.reserve(1);
    max_lostLayers_.reserve(1);
    min_hits_bypass_.reserve(1);
    applyAbsCutsIfNoPV_.reserve(1);
    max_d0NoPV_.reserve(1);
    max_z0NoPV_.reserve(1);
    preFilter_.reserve(1);
    max_relpterr_.reserve(1);
    min_nhits_.reserve(1);
    max_minMissHitOutOrIn_.reserve(1);
    max_lostHitFraction_.reserve(1);
    min_eta_.reserve(1);
    max_eta_.reserve(1);

    produces<edm::ValueMap<float> >("MVAVals");
    useAnyMVA_ = false;
    forest_ = 0;
    gbrVals_ = 0;  

    src_ = consumes<reco::TrackCollection>(cfg.getParameter<edm::InputTag>( "src" ));
    beamspot_ = consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>( "beamspot" ));
    useVertices_ = cfg.getParameter<bool>( "useVertices" );
    useVtxError_ = cfg.getParameter<bool>( "useVtxError" );
    if (useVertices_) vertices_ = consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>( "vertices" ));
    copyExtras_ = cfg.getUntrackedParameter<bool>("copyExtras", false);
    copyTrajectories_ = cfg.getUntrackedParameter<bool>("copyTrajectories", false);
    if (copyTrajectories_) {
        srcTraj_ = consumes<std::vector<Trajectory> >(cfg.getParameter<edm::InputTag>( "src" ));
        srcTass_ = consumes<TrajTrackAssociationCollection>(cfg.getParameter<edm::InputTag>( "src" ));
    }
    
    qualityToSet_.push_back( TrackBase::undefQuality );
    // parameters for vertex selection
    vtxNumber_.push_back( useVertices_ ? cfg.getParameter<int32_t>("vtxNumber") : 0 );
    vertexCut_.push_back( useVertices_ ? cfg.getParameter<std::string>("vertexCut") : "");
    //  parameters for adapted optimal cuts on chi2 and primary vertex compatibility
    res_par_.push_back(cfg.getParameter< std::vector<double> >("res_par") );
    chi2n_par_.push_back( cfg.getParameter<double>("chi2n_par") );
    chi2n_no1Dmod_par_.push_back( cfg.getParameter<double>("chi2n_no1Dmod_par") );
    d0_par1_.push_back(cfg.getParameter< std::vector<double> >("d0_par1"));
    dz_par1_.push_back(cfg.getParameter< std::vector<double> >("dz_par1"));
    d0_par2_.push_back(cfg.getParameter< std::vector<double> >("d0_par2"));
    dz_par2_.push_back(cfg.getParameter< std::vector<double> >("dz_par2"));

    // Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
    applyAdaptedPVCuts_.push_back(cfg.getParameter<bool>("applyAdaptedPVCuts"));
    // Impact parameter absolute cuts.
    max_d0_.push_back(cfg.getParameter<double>("max_d0"));
    max_z0_.push_back(cfg.getParameter<double>("max_z0"));
    nSigmaZ_.push_back(cfg.getParameter<double>("nSigmaZ"));
    // Cuts on numbers of layers with hits/3D hits/lost hits.
    min_layers_.push_back(cfg.getParameter<uint32_t>("minNumberLayers") );
    min_3Dlayers_.push_back(cfg.getParameter<uint32_t>("minNumber3DLayers") );
    max_lostLayers_.push_back(cfg.getParameter<uint32_t>("maxNumberLostLayers"));
    min_hits_bypass_.push_back(cfg.getParameter<uint32_t>("minHitsToBypassChecks"));
    max_relpterr_.push_back(cfg.getParameter<double>("max_relpterr"));
    min_nhits_.push_back(cfg.getParameter<uint32_t>("min_nhits"));
    max_minMissHitOutOrIn_.push_back(
	cfg.existsAs<int32_t>("max_minMissHitOutOrIn") ? 
	cfg.getParameter<int32_t>("max_minMissHitOutOrIn") : 99);
    max_lostHitFraction_.push_back(
	cfg.existsAs<double>("max_lostHitFraction") ?
	cfg.getParameter<double>("max_lostHitFraction") : 1.0);
    min_eta_.push_back(cfg.getParameter<double>("min_eta"));
    max_eta_.push_back(cfg.getParameter<double>("max_eta"));

    // Flag to apply absolute cuts if no PV passes the selection
    applyAbsCutsIfNoPV_.push_back(cfg.getParameter<bool>("applyAbsCutsIfNoPV"));
    keepAllTracks_.push_back( cfg.exists("keepAllTracks") ?
		              cfg.getParameter<bool>("keepAllTracks") :
		              false ); 
 
    setQualityBit_.push_back( false );
    std::string qualityStr = cfg.getParameter<std::string>("qualityBit");
    
    if(d0_par1_[0].size()!=2 || dz_par1_[0].size()!=2 || d0_par2_[0].size()!=2 || dz_par2_[0].size()!=2)
    {
      edm::LogError("MisConfiguration")<<"vector of size less then 2";
      throw; 
    }

    if (cfg.exists("qualityBit")) {
      std::string qualityStr = cfg.getParameter<std::string>("qualityBit");
      if (qualityStr != "") {
        setQualityBit_[0] = true;
        qualityToSet_ [0] = TrackBase::qualityByName(cfg.getParameter<std::string>("qualityBit"));
      }
    }
  
    if (keepAllTracks_[0] && !setQualityBit_[0]) throw cms::Exception("Configuration") << 
      "If you set 'keepAllTracks' to true, you must specify which qualityBit to set.\n";
    if (setQualityBit_[0] && (qualityToSet_[0] == TrackBase::undefQuality)) throw cms::Exception("Configuration") <<
      "You can't set the quality bit " << cfg.getParameter<std::string>("qualityBit") << " as it is 'undefQuality' or unknown.\n";
    if (applyAbsCutsIfNoPV_[0]) {
      max_d0NoPV_.push_back(cfg.getParameter<double>("max_d0NoPV"));
      max_z0NoPV_.push_back(cfg.getParameter<double>("max_z0NoPV"));
    }
    else{//dummy values
      max_d0NoPV_.push_back(0.);
      max_z0NoPV_.push_back(0.);
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
  evt.getByToken(beamspot_, hBsp);
  reco::BeamSpot vertexBeamSpot;
  vertexBeamSpot = *hBsp;
	
  // Select good primary vertices for use in subsequent track selection
  edm::Handle<reco::VertexCollection> hVtx;
  std::vector<Point> points;
  std::vector<float> vterr, vzerr;
  if (useVertices_) {
      evt.getByToken(vertices_, hVtx);
      selectVertices(0,*hVtx, points, vterr, vzerr);
      // Debug 
      LogDebug("SelectVertex") << points.size() << " good pixel vertices";
  }

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

  processMVA(evt,es);

  // Loop over tracks
  size_t current = 0;
  for (TrackCollection::const_iterator it = hSrcTrack->begin(), ed = hSrcTrack->end(); it != ed; ++it, ++current) {
    const Track & trk = * it;
    // Check if this track passes cuts

    LogTrace("TrackSelection") << "ready to check track with pt="<< trk.pt() ;

    double mvaVal = 0;
    if(useAnyMVA_)mvaVal = mvaVals_[current];
    bool ok = select(0,vertexBeamSpot, trk, points, vterr, vzerr,mvaVal);
    if (!ok) {

      LogTrace("TrackSelection") << "track with pt="<< trk.pt() << " NOT selected";

      if (copyTrajectories_) trackRefs_[current] = reco::TrackRef();
      if (!keepAllTracks_[0]) continue;
    }
    LogTrace("TrackSelection") << "track with pt="<< trk.pt() << " selected";
    selTracks_->push_back( Track( trk ) ); // clone and store
    if (ok && setQualityBit_[0]) {
      selTracks_->back().setQuality(qualityToSet_[0]);
      if (qualityToSet_[0]==TrackBase::tight) {
	selTracks_->back().setQuality(TrackBase::loose);
      } 
      else if (qualityToSet_[0]==TrackBase::highPurity) {
	selTracks_->back().setQuality(TrackBase::loose);
	selTracks_->back().setQuality(TrackBase::tight);
      }
      if (!points.empty()) {
	if (qualityToSet_[0]==TrackBase::loose) {
	  selTracks_->back().setQuality(TrackBase::looseSetWithPV);
	}
	else if (qualityToSet_[0]==TrackBase::highPurity) {
	  selTracks_->back().setQuality(TrackBase::looseSetWithPV);
	  selTracks_->back().setQuality(TrackBase::highPuritySetWithPV);
	}
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
    evt.getByToken(srcTass_, hTTAss);
    evt.getByToken(srcTraj_, hTraj);
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
