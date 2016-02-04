#include "RecoHI/HiTracking/interface/HiAnalyticalTrackSelector.h"

#include <Math/DistFunc.h>
#include "TMath.h"

using reco::modules::HiAnalyticalTrackSelector;

HiAnalyticalTrackSelector::HiAnalyticalTrackSelector( const edm::ParameterSet & cfg ) :
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
    max_relpterr_( cfg.getParameter<double>("max_relpterr") ),
    min_nhits_( cfg.getParameter<uint32_t>("min_nhits") ),
    vtxNumber_( cfg.getParameter<int32_t>("vtxNumber") ),
    vtxTracks_( cfg.getParameter<uint32_t>("vtxTracks") ),
    vtxChi2Prob_( cfg.getParameter<double>("vtxChi2Prob") ),
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
            produces< std::vector<Trajectory> >().setBranchAlias( alias + "Trajectories");
            produces< TrajTrackAssociationCollection >().setBranchAlias( alias + "TrajectoryTrackAssociations");
        }
 
}

HiAnalyticalTrackSelector::~HiAnalyticalTrackSelector() {
}

void HiAnalyticalTrackSelector::produce( edm::Event& evt, const edm::EventSetup& es ) 
{
    using namespace std; 
    using namespace edm;
    using namespace reco;

    Handle<TrackCollection> hSrcTrack;
    Handle< vector<Trajectory> > hTraj;
    Handle< vector<Trajectory> > hTrajP;
    Handle< TrajTrackAssociationCollection > hTTAss;

    bool isTrajThere = evt.getByLabel(src_, hTraj);

    // looking for the beam spot
    edm::Handle<reco::BeamSpot> hBsp;
    evt.getByLabel(beamspot_, hBsp);
    reco::BeamSpot vertexBeamSpot;
    vertexBeamSpot = *hBsp;
	
    // Select good primary vertices for use in subsequent track selection
    edm::Handle<reco::VertexCollection> hVtx;
    evt.getByLabel(vertices_, hVtx);
    std::vector<Point> points;
    std::vector<double> vterr, vzerr;
    selectVertices(*hVtx, points, vterr, vzerr);

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

    if (isTrajThere && copyTrajectories_) trackRefs_.resize(hSrcTrack->size());
    
    // Loop over tracks
    size_t current = 0;
    for (TrackCollection::const_iterator it = hSrcTrack->begin(), ed = hSrcTrack->end(); it != ed; ++it, ++current) {
        const Track & trk = * it;
	// Check if this track passes cuts
        bool ok = select(vertexBeamSpot, trk, points, vterr, vzerr);
        if (!ok) {
            if (isTrajThere && copyTrajectories_) trackRefs_[current] = reco::TrackRef();
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
            for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
                selHits_->push_back( (*hit)->clone() );
                tx.add( TrackingRecHitRef( rHits_, selHits_->size() - 1) );
            }
        }
        if (isTrajThere && copyTrajectories_) {
            trackRefs_[current] = TrackRef(rTracks_, selTracks_->size() - 1);
        }
    }
    if (isTrajThere && copyTrajectories_ ) {
       Handle< vector<Trajectory> > hTraj;
       Handle< TrajTrackAssociationCollection > hTTAss;
       evt.getByLabel(src_, hTTAss);
       evt.getByLabel(src_, hTraj); // it's gotten up there 
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
        if (isTrajThere &&  copyTrajectories_ ) {
            evt.put(selTrajs_);
            evt.put(selTTAss_);
        }
}


bool HiAnalyticalTrackSelector::select(const reco::BeamSpot &vertexBeamSpot, const reco::Track &tk, const std::vector<Point> &points, std::vector<double> &vterr, std::vector<double> &vzerr) {
  // Decide if the given track passes selection cuts.

   using namespace std; 

   // Cuts on numbers of layers with hits/3D hits/lost hits.
   uint32_t nlayers     = tk.hitPattern().trackerLayersWithMeasurement();
   uint32_t nlayers3D   = tk.hitPattern().pixelLayersWithMeasurement() +
                          tk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
   uint32_t nlayersLost = tk.hitPattern().trackerLayersWithoutMeasurement();
   if (nlayers < min_layers_) return false;
   if (nlayers3D < min_3Dlayers_) return false;
   if (nlayersLost > max_lostLayers_) return false;

   // Get track parameters
   double pt = tk.pt(),eta = tk.eta(), chi2n =  tk.normalizedChi2();
   double d0 = -tk.dxy(vertexBeamSpot.position()), d0E =  tk.d0Error(),
     dz = tk.dz(vertexBeamSpot.position()), dzE =  tk.dzError();

   double relpterr = tk.ptError()/pt;
   uint32_t nhits = tk.numberOfValidHits();

   if(relpterr > max_relpterr_) return false;
   if(nhits < min_nhits_) return false;

   // optimized cuts adapted to the track nlayers, pt, eta:
   // cut on chiquare/ndof 
   if (chi2n > chi2n_par_*nlayers) return false;


   // parametrized d0 resolution for the track pt
   double nomd0E = sqrt(res_par_[0]*res_par_[0]+(res_par_[1]/max(pt,1e-9))*(res_par_[1]/max(pt,1e-9)));
   // parametrized z0 resolution for the track pt and eta
   double nomdzE = nomd0E*(std::cosh(eta));


   // ---- PrimaryVertex compatibility cut
   bool primaryVertexZCompatibility(false);   
   bool primaryVertexD0Compatibility(false);   

   if (points.empty()) { //If not primaryVertices are reconstructed, check just the compatibility with the BS
     //z0 within three sigma of the beam spot z, if no good vertex is found
     if ( abs(dz) < (vertexBeamSpot.sigmaZ()*3) ) primaryVertexZCompatibility = true;  

     // d0 compatibility with beam line
     if (abs(d0) < pow(d0_par1_[0]*nlayers,d0_par1_[1])*nomd0E &&
	 abs(d0) < pow(d0_par2_[0]*nlayers,d0_par2_[1])*d0E) primaryVertexD0Compatibility = true;     
   }


   int iv=0;
   for (std::vector<Point>::const_iterator point = points.begin(), end = points.end(); point != end; ++point) {
     if(primaryVertexZCompatibility && primaryVertexD0Compatibility) break;
     double dzPV = tk.dz(*point); //re-evaluate the dz with respect to the vertex position
     double d0PV = tk.dxy(*point); //re-evaluate the dxy with respect to the vertex position

     /*
     if (abs(dzPV) < pow(dz_par1_[0]*nlayers,dz_par1_[1])*nomdzE &&
	 abs(dzPV) < pow(dz_par2_[0]*nlayers,dz_par2_[1])*dzE )  primaryVertexZCompatibility = true;

     if (abs(d0PV) < pow(d0_par1_[0]*nlayers,d0_par1_[1])*nomd0E &&
	 abs(d0PV) < pow(d0_par2_[0]*nlayers,d0_par2_[1])*d0E) primaryVertexD0Compatibility = true;   
     */

     // Edward's temporary changes
     double dzErrPV = sqrt(dzE*dzE+vzerr[iv]*vzerr[iv]);
     double d0ErrPV = sqrt(d0E*d0E+vterr[iv]*vterr[iv]);
     iv++;

     // Wei's temporary changes (plus Edward's max cuts)
     if (abs(dzPV) < dz_par1_[0]*pow(nlayers,dz_par1_[1])*nomdzE &&
         abs(dzPV) < dz_par2_[0]*pow(nlayers,dz_par2_[1])*dzErrPV &&
	 abs(dzPV) < max_z0_)  primaryVertexZCompatibility = true;
     
     if (abs(d0PV) < d0_par1_[0]*pow(nlayers,d0_par1_[1])*nomd0E &&
         abs(d0PV) < d0_par2_[0]*pow(nlayers,d0_par2_[1])*d0ErrPV &&
	 abs(d0PV) < max_d0_) primaryVertexD0Compatibility = true; 
     
  
   }


   // Absolute cuts on all tracks impact parameters with respect to beam-spot.
   // If BS is not compatible, verify if at least the reco-vertex is compatible (useful for incorrect BS settings)
   if (abs(d0) > max_d0_ && !primaryVertexD0Compatibility) return false;
   if (abs(dz) > max_z0_ && !primaryVertexZCompatibility) return false;


   if (applyAdaptedPVCuts_) {
     return (primaryVertexD0Compatibility && primaryVertexZCompatibility);
   } else {
     return true;     
   }


}
void HiAnalyticalTrackSelector::selectVertices(const reco::VertexCollection &vtxs, std::vector<Point> &points, std::vector<double> &vterr, std::vector<double> &vzerr) {
  // Select good primary vertices
    using namespace reco;
    int32_t toTake = vtxNumber_; 
    for (VertexCollection::const_iterator it = vtxs.begin(), ed = vtxs.end(); it != ed; ++it) {
        if ((it->tracksSize() >= vtxTracks_)  && 
                ( (it->chi2() == 0.0) || (TMath::Prob(it->chi2(), static_cast<int32_t>(it->ndof()) ) >= vtxChi2Prob_) ) ) {
            points.push_back(it->position()); 
	    vterr.push_back(sqrt(it->yError()*it->xError()));
	    vzerr.push_back(it->zError());
            toTake--; if (toTake == 0) break;
        }
    }
}

