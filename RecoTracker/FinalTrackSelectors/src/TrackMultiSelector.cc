#include "RecoTracker/FinalTrackSelectors/interface/TrackMultiSelector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <Math/DistFunc.h>
#include "TMath.h"

using reco::modules::TrackMultiSelector;
//using reco::modules::TrackMultiSelector::Block;

TrackMultiSelector::Block::Block(const edm::ParameterSet & cfg) :
    pt(p2p<double>(cfg,"pt")), 
    vlayers(p2p<uint32_t>(cfg,"validLayers")),
    lhits(p2p<uint32_t>(cfg,"lostHits")),
    chi2n(p2p<double>(cfg,"chi2n")), 
    d0(cfg.getParameter<double>("d0")),
    dz(cfg.getParameter<double>("dz")),
    d0Rel(cfg.getParameter<double>("d0Rel")),
    dzRel(cfg.getParameter<double>("dzRel"))
{
}

TrackMultiSelector::TrackMultiSelector( const edm::ParameterSet & cfg ) :
    src_( cfg.getParameter<edm::InputTag>( "src" ) ),
    vertices_( cfg.getParameter<edm::InputTag>( "vertices" ) ),
    copyExtras_(cfg.getUntrackedParameter<bool>("copyExtras", false)),
    copyTrajectories_(cfg.getUntrackedParameter<bool>("copyTrajectories", false)),
    splitOutputs_( cfg.getUntrackedParameter<bool>("splitOutputs", false) ),
    vtxNumber_( cfg.getParameter<int32_t>("vtxNumber") ),
    vtxTracks_( cfg.getParameter<uint32_t>("vtxTracks") ),
    vtxChi2Prob_( cfg.getParameter<double>("vtxChi2Prob") )
{
    edm::ParameterSet beamSpotPSet = cfg.getParameter<edm::ParameterSet>("beamspot");
    beamspot_   = beamSpotPSet.getParameter<edm::InputTag>("src");
    beamspotDZsigmas_ = beamSpotPSet.getParameter<double>("dzSigmas"); 
    beamspotD0_ = beamSpotPSet.getParameter<double>("d0"); 

    typedef std::vector<edm::ParameterSet> VPSet;
    VPSet psets = cfg.getParameter<VPSet>("cutSets");
    blocks_.reserve(psets.size());
    for (VPSet::const_iterator it = psets.begin(), ed = psets.end(); it != ed; ++it) {
        blocks_.push_back(TrackMultiSelector::Block(*it));
    }

    if (splitOutputs_) {
        char buff[15];
        for (size_t i = 0; i < blocks_.size(); ++i) {
            sprintf(buff,"set%d", static_cast<int>(i+1)); 
            labels_.push_back(std::string(buff));
        }
    } else {
        labels_.push_back(std::string(""));
    }

    std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
    for (std::vector<std::string>::const_iterator li = labels_.begin(), le = labels_.end(); li != le; ++li) {
        const char *l= li->c_str();
        produces<reco::TrackCollection>(l).setBranchAlias( alias + "Tracks" + l);
        if (copyExtras_) {
            produces<reco::TrackExtraCollection>(l).setBranchAlias( alias + "TrackExtras" + l);
            produces<TrackingRecHitCollection>(l).setBranchAlias( alias + "RecHits" + l);
            if (copyTrajectories_) {
                produces< std::vector<Trajectory> >(l).setBranchAlias( alias + "Trajectories" + l);
                produces< TrajTrackAssociationCollection >(l).setBranchAlias( alias + "TrajectoryTrackAssociations" + l);
            }
        }
    } 

    size_t nblocks = splitOutputs_ ? blocks_.size() : 1;
    selTracks_ = new std::auto_ptr<reco::TrackCollection>[nblocks];
    selTrackExtras_ = new std::auto_ptr<reco::TrackExtraCollection>[nblocks]; 
    selHits_ = new std::auto_ptr<TrackingRecHitCollection>[nblocks]; 
    selTrajs_ = new std::auto_ptr< std::vector<Trajectory> >[nblocks]; 
    selTTAss_ = new std::auto_ptr< TrajTrackAssociationCollection >[nblocks]; 
    rTracks_ = std::vector<reco::TrackRefProd>(nblocks);
    rHits_ = std::vector<TrackingRecHitRefProd>(nblocks);
    rTrackExtras_ = std::vector<reco::TrackExtraRefProd>(nblocks);
    rTrajectories_ = std::vector< edm::RefProd< std::vector<Trajectory> > >(nblocks);
    for (size_t i = 0; i < nblocks; i++) {
        selTracks_[i] = std::auto_ptr<reco::TrackCollection>(new reco::TrackCollection());
        selTrackExtras_[i] = std::auto_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection());
        selHits_[i] = std::auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
        selTrajs_[i] = std::auto_ptr< std::vector<Trajectory> >(new std::vector<Trajectory>()); 
        selTTAss_[i] = std::auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());
    }
 
}

TrackMultiSelector::~TrackMultiSelector() {
    delete [] selTracks_; 
    delete [] selTrackExtras_;
    delete [] selHits_;
    delete [] selTrajs_;
    delete [] selTTAss_;
}

void TrackMultiSelector::produce( edm::Event& evt, const edm::EventSetup& es ) 
{
    using namespace std; 
    using namespace edm;
    using namespace reco;

    size_t nblocks = splitOutputs_ ? blocks_.size() : 1;

    Handle<TrackCollection> hSrcTrack;
    Handle< vector<Trajectory> > hTraj;
    Handle< TrajTrackAssociationCollection > hTTAss;

    edm::Handle<reco::VertexCollection> hVtx;
    evt.getByLabel(vertices_, hVtx);
    std::vector<Point> points;
    if (vtxNumber_ != 0) selectVertices(*hVtx, points);

    edm::Handle<reco::BeamSpot> hBsp;
    evt.getByLabel(beamspot_, hBsp);

    evt.getByLabel( src_, hSrcTrack );

    for (size_t i = 0; i < nblocks; i++) {
        selTracks_[i] = auto_ptr<TrackCollection>(new TrackCollection());
        rTracks_[i] = evt.getRefBeforePut<TrackCollection>(labels_[i]);      
        if (copyExtras_) {
            selTrackExtras_[i] = auto_ptr<TrackExtraCollection>(new TrackExtraCollection());
            selHits_[i] = auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
            rHits_[i] = evt.getRefBeforePut<TrackingRecHitCollection>(labels_[i]);
            rTrackExtras_[i] = evt.getRefBeforePut<TrackExtraCollection>(labels_[i]);
        }
    }
    
    if (copyTrajectories_) whereItWent_.resize(hSrcTrack->size());
    size_t current = 0;
    for (TrackCollection::const_iterator it = hSrcTrack->begin(), ed = hSrcTrack->end(); it != ed; ++it, ++current) {
        const Track & trk = * it;
        short where = select(trk, *hBsp, points); 
        if (where == -1) {
            if (copyTrajectories_) whereItWent_[current] = std::pair<short, reco::TrackRef>(-1, reco::TrackRef());
            continue;
        }
        if (!splitOutputs_) where = 0;
        selTracks_[where]->push_back( Track( trk ) ); // clone and store
        if (!copyExtras_) continue;

        // TrackExtras
        selTrackExtras_[where]->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
                    trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
                    trk.outerStateCovariance(), trk.outerDetId(),
                    trk.innerStateCovariance(), trk.innerDetId(),
                    trk.seedDirection(), trk.seedRef() ) );
        selTracks_[where]->back().setExtra( TrackExtraRef( rTrackExtras_[where], selTrackExtras_[where]->size() - 1) );
        TrackExtra & tx = selTrackExtras_[where]->back();
	tx.setResiduals(trk.residuals());
        // TrackingRecHits
        for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
            selHits_[where]->push_back( (*hit)->clone() );
            tx.add( TrackingRecHitRef( rHits_[where], selHits_[where]->size() - 1) );
        }
        if (copyTrajectories_) {
            whereItWent_[current] = std::pair<short, reco::TrackRef>(where, TrackRef(rTracks_[where], selTracks_[where]->size() - 1));
        }
    }
    if ( copyTrajectories_ ) {
        Handle< vector<Trajectory> > hTraj;
        Handle< TrajTrackAssociationCollection > hTTAss;
        evt.getByLabel(src_, hTTAss);
        evt.getByLabel(src_, hTraj);
        for (size_t i = 0; i < nblocks; i++) {
            rTrajectories_[i] = evt.getRefBeforePut< vector<Trajectory> >(labels_[i]);
            selTrajs_[i] = auto_ptr< vector<Trajectory> >(new vector<Trajectory>()); 
            selTTAss_[i] = auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());
        }
        for (size_t i = 0, n = hTraj->size(); i < n; ++i) {
            Ref< vector<Trajectory> > trajRef(hTraj, i);
            TrajTrackAssociationCollection::const_iterator match = hTTAss->find(trajRef);
            if (match != hTTAss->end()) {
                const Ref<TrackCollection> &trkRef = match->val; 
                short oldKey = static_cast<short>(trkRef.key());
                if (whereItWent_[oldKey].first != -1) {
                    int where = whereItWent_[oldKey].first;
                    selTrajs_[where]->push_back( Trajectory(*trajRef) );
                    selTTAss_[where]->insert ( Ref< vector<Trajectory> >(rTrajectories_[where], selTrajs_[where]->size() - 1), whereItWent_[oldKey].second );
                }
            }
        }
    }


    static const std::string emptyString;
    for (size_t i = 0; i < nblocks; i++) {
        const std::string & lbl = ( splitOutputs_ ? labels_[i] : emptyString);
        evt.put(selTracks_[i], lbl);
        if (copyExtras_ ) {
            evt.put(selTrackExtras_[i], lbl); 
            evt.put(selHits_[i], lbl);
            if ( copyTrajectories_ ) {
                evt.put(selTrajs_[i], lbl);
                evt.put(selTTAss_[i], lbl);
            }
        }
    }
}

inline bool  TrackMultiSelector::testVtx ( const reco::Track &tk, const reco::BeamSpot &beamSpot,
					   const std::vector<Point> &points,
					   const TrackMultiSelector::Block &cut) {
    using std::abs;
    double d0Err =abs(tk.d0Error()), dzErr = abs(tk.dzError());  // not fully sure they're > 0!
    if (points.empty()) {
        Point spot = beamSpot.position();
        double dz = abs(tk.dz(spot)), d0 = abs(tk.dxy(spot));
        return ( dz < beamspotDZsigmas_*beamSpot.sigmaZ() ) && ( d0 < beamspotD0_ );
    }
    for (std::vector<Point>::const_iterator point = points.begin(), end = points.end(); point != end; ++point) {
        double dz = abs(tk.dz(*point)), d0 = abs(tk.dxy(*point));
        if ((dz < cut.dz) && (d0 < cut.d0) 
	    && fabs(dz/std::max(dzErr,1e-9)) < cut.dzRel && (d0/std::max(d0Err,1e-8) < cut.d0Rel )) return true;
    }
    return false;
}

short TrackMultiSelector::select(const reco::Track &tk, const reco::BeamSpot &beamSpot, const std::vector<Point> &points) {
   uint32_t vlayers = tk.hitPattern().trackerLayersWithMeasurement(), lhits = tk.numberOfLostHits();
   double pt = tk.pt(), chi2n =  tk.normalizedChi2();
   int which = 0;
   for (std::vector<TrackMultiSelector::Block>::const_iterator itb = blocks_.begin(), edb = blocks_.end(); itb != edb; ++itb, ++which) {
        if ( ( itb->vlayers.first <= vlayers ) && ( vlayers <= itb->vlayers.second ) &&
             ( itb->chi2n.first <= chi2n ) && ( chi2n <= itb->chi2n.second ) &&
             ( itb->pt.first    <= pt    ) && ( pt    <= itb->pt.second    ) &&
             ( itb->lhits.first <= lhits ) && ( lhits <= itb->lhits.second ) &&
             testVtx(tk, beamSpot, points, *itb) ) 
        {
            return which;
        }
    }
    return -1;
}
void TrackMultiSelector::selectVertices(const reco::VertexCollection &vtxs, std::vector<Point> &points) {
    using namespace reco;

    int32_t toTake = vtxNumber_; 
    for (VertexCollection::const_iterator it = vtxs.begin(), ed = vtxs.end(); it != ed; ++it) {
        if ((it->tracksSize() >= vtxTracks_)  && 
                ( (it->chi2() == 0.0) || (TMath::Prob(it->chi2(), static_cast<int32_t>(it->ndof()) ) >= vtxChi2Prob_) ) ) {
            points.push_back(it->position()); 
            toTake--; if (toTake == 0) break;
        }
    }
}
