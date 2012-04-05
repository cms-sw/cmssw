#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuon.h"

#include <cmath>

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

PropagateToMuon::PropagateToMuon(const edm::ParameterSet & iConfig) :
  useSimpleGeometry_(iConfig.getParameter<bool>("useSimpleGeometry")),
  useMB2_(iConfig.existsAs<bool>("useStation2") ? iConfig.getParameter<bool>("useStation2") : true),
  fallbackToME1_(iConfig.existsAs<bool>("fallbackToME1") ? iConfig.getParameter<bool>("fallbackToME1") : false),
  whichTrack_(None), whichState_(AtVertex),
  cosmicPropagation_(iConfig.existsAs<bool>("cosmicPropagationHypothesis") ? iConfig.getParameter<bool>("cosmicPropagationHypothesis") : false)
{
    std::string whichTrack = iConfig.getParameter<std::string>("useTrack");
    if      (whichTrack == "none")    { whichTrack_ = None; }
    else if (whichTrack == "tracker") { whichTrack_ = TrackerTk; }
    else if (whichTrack == "muon")    { whichTrack_ = MuonTk; }
    else if (whichTrack == "global")  { whichTrack_ = GlobalTk; }
    else throw cms::Exception("Configuration") << "Parameter 'useTrack' must be 'none', 'tracker', 'muon', 'global'\n";
    if (whichTrack_ != None) {
        std::string whichState = iConfig.getParameter<std::string>("useState");
        if      (whichState == "atVertex")  { whichState_ = AtVertex; }
        else if (whichState == "innermost") { whichState_ = Innermost; }
        else if (whichState == "outermost") { whichState_ = Outermost; }
        else throw cms::Exception("Configuration") << "Parameter 'useState' must be 'atVertex', 'innermost', 'outermost'\n";
    }
    if (cosmicPropagation_ && (whichTrack_ == None || whichState_ == AtVertex)) {
        throw cms::Exception("Configuration") << "When using 'cosmicPropagationHypothesis' useTrack must not be 'none', and the state must not be 'atVertex'\n";
    }
}

PropagateToMuon::~PropagateToMuon() {}

void
PropagateToMuon::init(const edm::EventSetup & iSetup) {
    iSetup.get<IdealMagneticFieldRecord>().get(magfield_);
    iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAlong",    propagator_);
    iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorOpposite", propagatorOpposite_);
    iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny",      propagatorAny_);
    iSetup.get<MuonRecoGeometryRecord>().get(muonGeometry_);

    // Get the barrel cylinder
    const DetLayer * dtLay = muonGeometry_->allDTLayers()[useMB2_ ? 1 : 0];
    barrelCylinder_ = dynamic_cast<const BoundCylinder *>(&dtLay->surface());
    barrelHalfLength_ = barrelCylinder_->bounds().length()/2;;
    //std::cout << "L1MuonMatcher: barrel radius = " << barrelCylinder_->radius() << ", half length = " << barrelHalfLength_ << std::endl;

    // Get the endcap disks. Note that ME1 has two disks (ME1/1 and ME2/1-ME3/2-ME4/1), so there's one more index
    for (size_t i = 0; i <= (useMB2_ ? 2 : 1); ++i) {
        endcapDiskPos_[i] = dynamic_cast<const BoundDisk *>(& muonGeometry_->forwardCSCLayers()[i]->surface());
        endcapDiskNeg_[i] = dynamic_cast<const BoundDisk *>(& muonGeometry_->backwardCSCLayers()[i]->surface());
        endcapRadii_[i] = std::make_pair(endcapDiskPos_[i]->innerRadius(), endcapDiskPos_[i]->outerRadius());
        //std::cout << "L1MuonMatcher: endcap " << i << " Z = " << endcapDiskPos_[i]->position().z() << ", radii = " << endcapRadii_[i].first << "," << endcapRadii_[i].second << std::endl;
    }
}

FreeTrajectoryState 
PropagateToMuon::startingState(const reco::Candidate &reco) const {
    FreeTrajectoryState ret;
    if (whichTrack_ != None) {
        const reco::RecoCandidate *rc = dynamic_cast<const reco::RecoCandidate *>(&reco);
        if (rc == 0) throw cms::Exception("Invalid Data") << "Input object is not a RecoCandidate.\n";
        reco::TrackRef tk;
        switch (whichTrack_) {
            case TrackerTk: tk = rc->track();          break; 
            case MuonTk   : tk = rc->standAloneMuon(); break;
            case GlobalTk : tk = rc->combinedMuon();   break;
            default: break; // just to make gcc happy
        }
        if (tk.isNull()) {
            ret = FreeTrajectoryState();
        } else {
            ret = startingState(*tk);
        }
    } else {
        ret = FreeTrajectoryState(  GlobalPoint( reco.vx(), reco.vy(), reco.vz()),
                                    GlobalVector(reco.px(), reco.py(), reco.pz()),
                                    reco.charge(),
                                    magfield_.product());
    }
    return ret;
}

FreeTrajectoryState 
PropagateToMuon::startingState(const reco::Track &tk) const {
    if (!magfield_.isValid()) throw cms::Exception("NotInitialized") << "PropagateToMuon: You must call init(const edm::EventSetup &iSetup) before using this object.\n"; 
    WhichState state = whichState_;
    if (cosmicPropagation_) { 
        if (whichState_ == Innermost) {
            state = tk.innerPosition().Mag2() <= tk.outerPosition().Mag2()  ? Innermost : Outermost;
        } else if (whichState_ == Outermost) {
            state = tk.innerPosition().Mag2() <= tk.outerPosition().Mag2()  ? Outermost : Innermost;
        }
    }
    switch (state) {
        case Innermost: return trajectoryStateTransform::innerFreeState(  tk, magfield_.product()); 
        case Outermost: return trajectoryStateTransform::outerFreeState(  tk, magfield_.product()); 

        case AtVertex:  
        default:
            return trajectoryStateTransform::initialFreeState(tk, magfield_.product()); 
    }
    
}

FreeTrajectoryState 
PropagateToMuon::startingState(const SimTrack &tk, const edm::SimVertexContainer &vtxs) const {
    if (!magfield_.isValid()) throw cms::Exception("NotInitialized") << "PropagateToMuon: You must call init(const edm::EventSetup &iSetup) before using this object.\n"; 
    if (tk.noVertex()) throw cms::Exception("UnsupportedOperation") << "I can't propagate simtracks without a vertex, I don't know where to start from.\n";
    const math::XYZTLorentzVectorD & vtx = (vtxs)[tk.vertIndex()].position();
    return FreeTrajectoryState(  GlobalPoint(vtx.X(), vtx.Y(), vtx.Z()),
                                 GlobalVector(tk.momentum().X(), tk.momentum().Y(), tk.momentum().Z()),
                                 int(tk.charge()),
                                 magfield_.product());
}



TrajectoryStateOnSurface
PropagateToMuon::extrapolate(const FreeTrajectoryState &start) const {
    if (!magfield_.isValid() || barrelCylinder_ == 0) {
        throw cms::Exception("NotInitialized") << "PropagateToMuon: You must call init(const edm::EventSetup &iSetup) before using this object.\n"; 
    }

    TrajectoryStateOnSurface final;
    if (start.momentum().mag() == 0) return final;
    double eta = start.momentum().eta();

    const Propagator * propagatorBarrel  = &*propagator_;
    const Propagator * propagatorEndcaps = &*propagator_;
    if (whichState_ != AtVertex) { 
        if (start.position().perp()    > barrelCylinder_->radius())         propagatorBarrel  = &*propagatorOpposite_;
        if (fabs(start.position().z()) > endcapDiskPos_[useMB2_?2:1]->position().z()) propagatorEndcaps = &*propagatorOpposite_;
    }
    if (cosmicPropagation_) {
        if (start.momentum().dot(GlobalVector(start.position().x(), start.position().y(), start.position().z())) < 0) {
            // must flip the propagations
            propagatorBarrel  = (propagatorBarrel  == &*propagator_ ? &*propagatorOpposite_ : &*propagator_);
            propagatorEndcaps = (propagatorEndcaps == &*propagator_ ? &*propagatorOpposite_ : &*propagator_);
        }
    }

    TrajectoryStateOnSurface tsos = propagatorBarrel->propagate(start, *barrelCylinder_);
    if (tsos.isValid()) {
        if (useSimpleGeometry_) {
            //std::cout << "  propagated to barrel, z = " << tsos.globalPosition().z() << ", bound = " << barrelHalfLength_ << std::endl;
            if (fabs(tsos.globalPosition().z()) <= barrelHalfLength_) final = tsos;
        } else {
            final = getBestDet(tsos, muonGeometry_->allDTLayers()[1]);
        }
    } 
    if (!final.isValid()) { 
      for (int ie = (useMB2_ ? 2 : 1); ie >= 0; --ie) {
        tsos = propagatorEndcaps->propagate(start, (eta > 0 ? *endcapDiskPos_[ie] : *endcapDiskNeg_[ie]));
        if (tsos.isValid()) {
            if (useSimpleGeometry_) {
                float rho = tsos.globalPosition().perp();
                //std::cout << "  propagated to endcap " << ie << ", rho = " << rho << ", bounds [ " << endcapRadii_[ie].first << ", " << endcapRadii_[ie].second << "]" << std::endl;
                if ((rho >= endcapRadii_[ie].first) && (rho <= endcapRadii_[ie].second)) final = tsos;
            } else {
                final = getBestDet(tsos, (eta > 0 ? muonGeometry_->forwardCSCLayers()[ie] : muonGeometry_->backwardCSCLayers()[ie]));
            }
        } //else //std::cout << "  failed to propagated to endcap " << ie  << std::endl;
        if (final.isValid()) break;
        if (ie == 2 && !fallbackToME1_) break;
      }
    }
    return final;
}

TrajectoryStateOnSurface 
PropagateToMuon::getBestDet(const TrajectoryStateOnSurface &tsos, const DetLayer *layer) const {
    TrajectoryStateOnSurface ret; // start as null
    Chi2MeasurementEstimator estimator(1e10, 3.); // require compatibility at 3 sigma
    std::vector<GeometricSearchDet::DetWithState> dets = layer->compatibleDets(tsos, *propagatorAny_, estimator);
    if (!dets.empty()) {
        ret = dets.front().second;
    }
    return ret;
}
