#include "HLTriggerOffline/Muon/interface/PropagateToMuon.h"

#include <iostream>
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

    const DetLayer * dt2 = muonGeometry_->allDTLayers()[1];
    const DetLayer * csc2Pos = muonGeometry_->forwardCSCLayers()[2];
    const DetLayer * csc2Neg = muonGeometry_->backwardCSCLayers()[2];
    barrelCylinder_ = dynamic_cast<const BoundCylinder *>(&dt2->surface());
    endcapDiskPos_  = dynamic_cast<const BoundDisk *>(& csc2Pos->surface());
    endcapDiskNeg_  = dynamic_cast<const BoundDisk *>(& csc2Neg->surface());
    if (barrelCylinder_==nullptr || endcapDiskPos_==nullptr || endcapDiskNeg_==nullptr) throw cms::Exception("Geometry") << "Bad muon geometry!?";
    barrelHalfLength_ = barrelCylinder_->bounds().length()/2;;
    endcapRadii_ = std::make_pair(endcapDiskPos_->innerRadius(), endcapDiskPos_->outerRadius());
    //std::cout << "L1MuonMatcher: barrel radius = " << barrelCylinder_->radius() << ", half length = " << barrelHalfLength_ <<
    //             "; endcap Z = " << endcapDiskPos_->position().z() << ", radii = " << endcapRadii_.first << "," << endcapRadii_.second << std::std::endl;
}

FreeTrajectoryState 
PropagateToMuon::startingState(const reco::Candidate &reco) const {
    FreeTrajectoryState ret;
    if (whichTrack_ != None) {
        const reco::RecoCandidate *rc = dynamic_cast<const reco::RecoCandidate *>(&reco);
        if (rc == nullptr) throw cms::Exception("Invalid Data") << "Input object is not a RecoCandidate.\n";
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


TrajectoryStateOnSurface
PropagateToMuon::extrapolate(const FreeTrajectoryState &start) const {
    TrajectoryStateOnSurface final;
    if (start.momentum().mag() == 0) return final;
    double eta = start.momentum().eta();

    const Propagator * propagatorBarrel  = &*propagator_;
    const Propagator * propagatorEndcaps = &*propagator_;

    if (whichState_ != AtVertex) { 
        if (start.position().perp()    > barrelCylinder_->radius())      propagatorBarrel  = &*propagatorOpposite_;
        if (fabs(start.position().z()) > endcapDiskPos_->position().z()) propagatorEndcaps = &*propagatorOpposite_;
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
            if (fabs(tsos.globalPosition().z()) <= barrelHalfLength_) final = tsos;
        } else {
            final = getBestDet(tsos, muonGeometry_->allDTLayers()[1]);
        }
    } 

    if (!final.isValid()) { 
        tsos = propagatorEndcaps->propagate(start, (eta > 0 ? *endcapDiskPos_ : *endcapDiskNeg_));
        if (tsos.isValid()) {
            if (useSimpleGeometry_) {
                float rho = tsos.globalPosition().perp();
                if ((rho >= endcapRadii_.first) && (rho <= endcapRadii_.second)) final = tsos;
            } else {
                final = getBestDet(tsos, (eta > 0 ? muonGeometry_->forwardCSCLayers()[2] : muonGeometry_->backwardCSCLayers()[2]));
            }
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
