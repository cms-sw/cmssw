#include "MuonAnalysis/MuonAssociators/interface/MatcherUsingTracksAlgorithm.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// a few template-related workarounds
template<>
inline double deltaR<GlobalVector>(const GlobalVector &v1, const GlobalVector &v2) {
    return deltaR<float>(v1.eta(),v1.phi(),v2.eta(),v2.phi());
}
template<>
inline double deltaR2<GlobalVector>(const GlobalVector &v1, const GlobalVector &v2) {
    return deltaR2<float>(v1.eta(),v1.phi(),v2.eta(),v2.phi());
}

MatcherUsingTracksAlgorithm::MatcherUsingTracksAlgorithm(const edm::ParameterSet & iConfig) :
    whichTrack1_(None),     whichTrack2_(None), 
    whichState1_(AtVertex), whichState2_(AtVertex)
{
    std::string algo = iConfig.getParameter<std::string>("algorithm");
    if      (algo == "byTrackRef")           { algo_ = ByTrackRef; }
    else if (algo == "byPropagatingSrc")     { algo_ = ByPropagatingSrc; }
    else if (algo == "byPropagatingMatched") { algo_ = ByPropagatingMatched; }
    else if (algo == "byDirectComparison")   { algo_ = ByDirectComparison; }
    else throw cms::Exception("Configuration") << "Value '" << algo << "' for algorithm not yet implemented.\n";

    getConf(iConfig, "src",     whichTrack1_, whichState1_);
    getConf(iConfig, "matched", whichTrack2_, whichState2_);

    if (algo_ == ByTrackRef) {
        // validate the config
        if (whichTrack1_ == None || whichTrack2_ == None) throw cms::Exception("Configuration") << "Algorithm 'byTrackRef' needs tracks not to be 'none'.\n";
    } else if (algo_ == ByPropagatingSrc || algo_ == ByPropagatingMatched || algo_ == ByDirectComparison) {
        // read matching cuts
        maxLocalPosDiff_    = iConfig.getParameter<double>("maxDeltaLocalPos");
        maxGlobalMomDeltaR_ = iConfig.getParameter<double>("maxDeltaR");
        maxGlobalDPtRel_    = iConfig.getParameter<double>("maxDeltaPtRel");
        // choice of sorting variable
        std::string sortBy = iConfig.getParameter<std::string>("sortBy");
        if      (sortBy == "deltaLocalPos") sortBy_ = LocalPosDiff;
        else if (sortBy == "deltaPtRel")    sortBy_ = GlobalDPtRel;
        else if (sortBy == "deltaR")        sortBy_ = GlobalMomDeltaR;
        else throw cms::Exception("Configuration") << "Parameter 'sortBy' must be one of: deltaLocalPos, deltaPtRel, deltaR.\n";
        // validate the config
        if (algo_ == ByPropagatingSrc) {
            if (whichTrack2_ == None || whichState2_ == AtVertex) {
                throw cms::Exception("Configuration") << "Destination track must be non-null, and state must not be 'AtVertex' (not yet).\n";
            }
        } else if (algo_ == ByPropagatingMatched) {
            if (whichTrack1_ == None || whichState1_ == AtVertex) {
                throw cms::Exception("Configuration") << "Destination track must be non-null, and state must not be 'AtVertex' (not yet).\n";
            }
        } else if (algo_ == ByDirectComparison) {
            bool firstAtVertex = (whichTrack1_ == None || whichState1_ == AtVertex);
            bool   secAtVertex = (whichTrack2_ == None || whichState2_ == AtVertex);
            if (firstAtVertex) {
                if (!secAtVertex)  throw cms::Exception("Configuration") << "When using 'byDirectComparison' with 'src' at vertex (or None), 'matched' must be at vertex too.\n";
            } else {
                if (secAtVertex)  throw cms::Exception("Configuration") << "When using 'byDirectComparison' with 'src' not at vertex, 'matched' can't be at vertex or None.\n";
                if (whichState1_ != whichState2_) throw cms::Exception("Configuration") << "You can't use 'byDirectComparison' with non-matching states.\n";
            }
        }
    }
}

void 
MatcherUsingTracksAlgorithm::getConf(const edm::ParameterSet & iConfig, const std::string &whatFor, WhichTrack &whichTrack, WhichState &whichState) {
    std::string s_whichTrack = iConfig.getParameter<std::string>(whatFor+"Track");
    if      (s_whichTrack == "none")    { whichTrack = None; }
    else if (s_whichTrack == "tracker") { whichTrack = TrackerTk; }
    else if (s_whichTrack == "muon")    { whichTrack = MuonTk; }
    else if (s_whichTrack == "global")  { whichTrack = GlobalTk; }
    else throw cms::Exception("Configuration") << "Parameter 'useTrack' must be 'none', 'tracker', 'muon', 'global'\n";
    if ((whichTrack != None) && (algo_ != ByTrackRef)) {
        std::string s_whichState = iConfig.getParameter<std::string>(whatFor+"State");
        if      (s_whichState == "atVertex")  { whichState = AtVertex; }
        else if (s_whichState == "innermost") { whichState = Innermost; }
        else if (s_whichState == "outermost") { whichState = Outermost; }
        else throw cms::Exception("Configuration") << "Parameter 'useState' must be 'atVertex', 'innermost', 'outermost'\n";
    }
}

/// Try to match one track to another one. Return true if succeeded.
/// For matches not by ref, it will update deltaR, deltaLocalPos and deltaPtRel if the match suceeded
bool 
MatcherUsingTracksAlgorithm::match(const reco::Candidate &c1, const reco::Candidate &c2, float &deltR, float &deltaLocalPos, float &deltaPtRel) const {
    switch (algo_) {
        case ByTrackRef: { 
            reco::TrackRef t1 = getTrack(c1, whichTrack1_); 
            reco::TrackRef t2 = getTrack(c2, whichTrack2_); 
            if (t1.isNonnull()) {
                if (t1 == t2) return true; 
                if (t1.id() != t2.id()) { edm::LogWarning("MatcherUsingTracksAlgorithm") << "Trying to match by reference tracks coming from different collections.\n"; }
            } 
            }
        case ByPropagatingSrc: {
            FreeTrajectoryState start = startingState(c1, whichTrack1_, whichState1_);
            TrajectoryStateOnSurface target = targetState(c2, whichTrack2_, whichState2_);
            return matchWithPropagation(start, target, deltR, deltaLocalPos, deltaPtRel);
            }
        case ByPropagatingMatched: { 
            FreeTrajectoryState start = startingState(c2, whichTrack2_, whichState2_);
            TrajectoryStateOnSurface target = targetState(c1, whichTrack1_, whichState1_);
            return matchWithPropagation(start, target, deltR, deltaLocalPos, deltaPtRel);
            }
        case ByDirectComparison: {
            FreeTrajectoryState start = startingState(c1, whichTrack1_, whichState1_);
            FreeTrajectoryState otherstart = startingState(c2, whichTrack2_, whichState2_);
            return matchByDirectComparison(start, otherstart, deltR, deltaLocalPos, deltaPtRel);
            }
    }
    return false;
}

/// Find the best match to another candidate, and return its index in the vector
/// For matches not by ref, it will update deltaR, deltaLocalPos and deltaPtRel if the match suceeded
/// Returns -1 if the match fails
int 
MatcherUsingTracksAlgorithm::match(const reco::Candidate &c1, const edm::View<reco::Candidate> &c2s, float &deltR, float &deltaLocalPos, float &deltaPtRel) const {
    
    // working and output variables
    FreeTrajectoryState start; TrajectoryStateOnSurface target;
    int match = -1;

    // pre-fetch some states if needed
    if (algo_ == ByPropagatingSrc || algo_ == ByDirectComparison) start = startingState(c1, whichTrack1_, whichState1_);
    else if (algo_ == ByPropagatingMatched) target = targetState(c1, whichTrack1_, whichState1_);

    // loop on the  collection
    edm::View<reco::Candidate>::const_iterator it, ed; int i;
    for (it = c2s.begin(), ed = c2s.end(), i = 0; it != ed; ++it, ++i) {
        bool exit = false;
        switch (algo_) {
            case ByTrackRef: {
                     reco::TrackRef t1 = getTrack(c1, whichTrack1_); 
                     reco::TrackRef t2 = getTrack(*it, whichTrack2_); 
                     if (t1.isNonnull()) {
                         if (t1 == t2) { match = i; exit = true; }
                         if (t1.id() != t2.id()) { edm::LogWarning("MatcherUsingTracksAlgorithm") << "Trying to match by reference tracks coming from different collections.\n"; }
                     } 
                 } break;
            case ByPropagatingSrc: 
            case ByPropagatingMatched: {
                     if (algo_ == ByPropagatingMatched)  start = startingState(*it, whichTrack2_, whichState2_);
                     else if (algo_ == ByPropagatingSrc) target = targetState(*it, whichTrack2_, whichState2_);
                     if (matchWithPropagation(start, target, deltR, deltaLocalPos, deltaPtRel)) { 
                         match = i; 
                     }
                 } break;
            case ByDirectComparison: {
                     FreeTrajectoryState otherstart = startingState(*it, whichTrack2_, whichState2_);
                     if (matchByDirectComparison(start, otherstart, deltR, deltaLocalPos, deltaPtRel)) {
                         match = i;
                     }
                 } break;
        }
        if (exit) break;
    }

    return match;
}

void 
MatcherUsingTracksAlgorithm::init(const edm::EventSetup & iSetup) {
    iSetup.get<IdealMagneticFieldRecord>().get(magfield_);
    iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", propagator_);
    iSetup.get<GlobalTrackingGeometryRecord>().get(geometry_);
}


reco::TrackRef
MatcherUsingTracksAlgorithm::getTrack(const reco::Candidate &reco, WhichTrack whichTrack) const {
    reco::TrackRef tk;
    const reco::RecoCandidate *rc = dynamic_cast<const reco::RecoCandidate *>(&reco);
    if (rc == 0) throw cms::Exception("Invalid Data") << "Input object is not a RecoCandidate.\n";
    switch (whichTrack) {
        case TrackerTk: tk = rc->track();          break; 
        case MuonTk   : tk = rc->standAloneMuon(); break;
        case GlobalTk : tk = rc->combinedMuon();   break;
        default: break; // just to make gcc happy
    }
    return tk;
}

FreeTrajectoryState 
MatcherUsingTracksAlgorithm::startingState(const reco::Candidate &reco, WhichTrack whichTrack, WhichState whichState) const {
    FreeTrajectoryState ret;
    if (whichTrack != None) {
        reco::TrackRef tk = getTrack(reco, whichTrack);
        if (tk.isNull()) {
            ret = FreeTrajectoryState();
        } else {
            switch (whichState) {
                case AtVertex:  ret = TrajectoryStateTransform().initialFreeState(*tk, magfield_.product()); break;
                case Innermost: ret = TrajectoryStateTransform().innerFreeState(  *tk, magfield_.product()); break;
                case Outermost: ret = TrajectoryStateTransform().outerFreeState(  *tk, magfield_.product()); break;
            }
        }
    } else {
        ret = FreeTrajectoryState(  GlobalPoint( reco.vx(), reco.vy(), reco.vz()),
                                    GlobalVector(reco.px(), reco.py(), reco.pz()),
                                    reco.charge(),
                                    magfield_.product());
    }
    return ret;
}

TrajectoryStateOnSurface 
MatcherUsingTracksAlgorithm::targetState(const reco::Candidate &reco, WhichTrack whichTrack, WhichState whichState) const {
    TrajectoryStateOnSurface ret;
    reco::TrackRef tk = getTrack(reco, whichTrack);
    if (tk.isNonnull()) {
        switch (whichState) {
            case Innermost: ret = TrajectoryStateTransform().innerStateOnSurface(  *tk, *geometry_, magfield_.product()); break;
            case Outermost: ret = TrajectoryStateTransform().outerStateOnSurface(  *tk, *geometry_, magfield_.product()); break;
            default: break; // just to make gcc happy
        }
    }
    return ret;
}

bool
MatcherUsingTracksAlgorithm::matchWithPropagation(const FreeTrajectoryState &start, 
                                              const TrajectoryStateOnSurface &target, 
                                              float &lastDeltaR, 
                                              float &lastDeltaLocalPos,
                                              float &lastGlobalDPtRel) const 
{
    if ((start.momentum().mag() == 0) || !target.isValid()) return false;

    TrajectoryStateOnSurface tsos = propagator_->propagate(start, target.surface());

    bool isBest = false;
    if (tsos.isValid()) {
        float thisLocalPosDiff = (tsos.localPosition()-target.localPosition()).mag();
        float thisGlobalMomDeltaR = deltaR(tsos.globalMomentum(), target.globalMomentum());
        float thisGlobalDPtRel = (tsos.globalMomentum().perp() - target.globalMomentum().perp())/target.globalMomentum().perp();

        if ((thisLocalPosDiff       < maxLocalPosDiff_) &&
            (thisGlobalMomDeltaR    < maxGlobalMomDeltaR_) &&
            (fabs(thisGlobalDPtRel) < maxGlobalDPtRel_)) {
            switch (sortBy_) {
                case LocalPosDiff:    isBest = (thisLocalPosDiff    < lastDeltaLocalPos); break;
                case GlobalMomDeltaR: isBest = (thisGlobalMomDeltaR < lastDeltaR);        break;
                case GlobalDPtRel:    isBest = (thisGlobalDPtRel    < lastGlobalDPtRel);  break;
            }
            if (isBest) {
                lastDeltaLocalPos = thisLocalPosDiff;
                lastDeltaR        = thisGlobalMomDeltaR;
                lastGlobalDPtRel  = thisGlobalDPtRel;
            } 
        }  // if match
    }

    return isBest;
}

bool
MatcherUsingTracksAlgorithm::matchByDirectComparison(const FreeTrajectoryState &start, 
                                                 const FreeTrajectoryState &target, 
                                                 float &lastDeltaR, 
                                                 float &lastDeltaLocalPos,
                                                 float &lastGlobalDPtRel) const 
{
    if ((start.momentum().mag() == 0) || target.momentum().mag() == 0) return false;

    bool isBest = false;
    float thisLocalPosDiff = (start.position()-target.position()).mag();
    float thisGlobalMomDeltaR = deltaR(start.momentum(), target.momentum());
    float thisGlobalDPtRel = (start.momentum().perp() - target.momentum().perp())/target.momentum().perp();

    if ((thisLocalPosDiff       < maxLocalPosDiff_) &&
            (thisGlobalMomDeltaR    < maxGlobalMomDeltaR_) &&
            (fabs(thisGlobalDPtRel) < maxGlobalDPtRel_)) {
        switch (sortBy_) {
            case LocalPosDiff:    isBest = (thisLocalPosDiff    < lastDeltaLocalPos); break;
            case GlobalMomDeltaR: isBest = (thisGlobalMomDeltaR < lastDeltaR);        break;
            case GlobalDPtRel:    isBest = (thisGlobalDPtRel    < lastGlobalDPtRel);  break;
        }
        if (isBest) {
            lastDeltaLocalPos = thisLocalPosDiff;
            lastDeltaR        = thisGlobalMomDeltaR;
            lastGlobalDPtRel  = thisGlobalDPtRel;
        } 
    }  // if match

    return isBest;
}
