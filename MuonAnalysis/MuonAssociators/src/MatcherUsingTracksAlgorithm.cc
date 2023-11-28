#include "MuonAnalysis/MuonAssociators/interface/MatcherUsingTracksAlgorithm.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

MatcherUsingTracksAlgorithm::MatcherUsingTracksAlgorithm(const edm::ParameterSet &iConfig, edm::ConsumesCollector iC)
    : whichTrack1_(None),
      whichTrack2_(None),
      whichState1_(AtVertex),
      whichState2_(AtVertex),
      srcCut_(iConfig.existsAs<std::string>("srcPreselection") ? iConfig.getParameter<std::string>("srcPreselection")
                                                               : ""),
      matchedCut_(iConfig.existsAs<std::string>("matchedPreselection")
                      ? iConfig.getParameter<std::string>("matchedPreselection")
                      : ""),
      requireSameCharge_(iConfig.existsAs<bool>("requireSameCharge") ? iConfig.getParameter<bool>("requireSameCharge")
                                                                     : false),
      magfieldToken_(iC.esConsumes()),
      propagatorToken_(iC.esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorAny"))),
      geometryToken_(iC.esConsumes()) {
  std::string algo = iConfig.getParameter<std::string>("algorithm");
  if (algo == "byTrackRef") {
    algo_ = ByTrackRef;
  } else if (algo == "byPropagatingSrc") {
    algo_ = ByPropagatingSrc;
  } else if (algo == "byPropagatingMatched") {
    algo_ = ByPropagatingMatched;
  } else if (algo == "byDirectComparison") {
    algo_ = ByDirectComparison;
  } else
    throw cms::Exception("Configuration") << "Value '" << algo << "' for algorithm not yet implemented.\n";

  getConf(iConfig, "src", whichTrack1_, whichState1_);
  getConf(iConfig, "matched", whichTrack2_, whichState2_);

  if (algo_ == ByTrackRef) {
    // validate the config
    if (whichTrack1_ == None || whichTrack2_ == None)
      throw cms::Exception("Configuration") << "Algorithm 'byTrackRef' needs tracks not to be 'none'.\n";
  } else if (algo_ == ByPropagatingSrc || algo_ == ByPropagatingMatched || algo_ == ByDirectComparison) {
    // read matching cuts
    maxLocalPosDiff_ = iConfig.getParameter<double>("maxDeltaLocalPos");
    maxGlobalMomDeltaR_ = iConfig.getParameter<double>("maxDeltaR");
    maxGlobalMomDeltaEta_ =
        iConfig.existsAs<double>("maxDeltaEta") ? iConfig.getParameter<double>("maxDeltaEta") : maxGlobalMomDeltaR_;
    maxGlobalMomDeltaPhi_ =
        iConfig.existsAs<double>("maxDeltaPhi") ? iConfig.getParameter<double>("maxDeltaPhi") : maxGlobalMomDeltaR_;
    maxGlobalDPtRel_ = iConfig.getParameter<double>("maxDeltaPtRel");

    // choice of sorting variable
    std::string sortBy = iConfig.getParameter<std::string>("sortBy");
    if (sortBy == "deltaLocalPos")
      sortBy_ = LocalPosDiff;
    else if (sortBy == "deltaPtRel")
      sortBy_ = GlobalDPtRel;
    else if (sortBy == "deltaR")
      sortBy_ = GlobalMomDeltaR;
    else if (sortBy == "deltaEta")
      sortBy_ = GlobalMomDeltaEta;
    else if (sortBy == "deltaPhi")
      sortBy_ = GlobalMomDeltaPhi;
    else if (sortBy == "chi2")
      sortBy_ = Chi2;
    else
      throw cms::Exception("Configuration")
          << "Parameter 'sortBy' must be one of: deltaLocalPos, deltaPtRel, deltaR, chi2.\n";
    // validate the config
    if (algo_ == ByPropagatingSrc) {
      if (whichTrack2_ == None || whichState2_ == AtVertex) {
        algo_ = ByPropagatingSrcTSCP;
        //throw cms::Exception("Configuration") << "Destination track must be non-null, and state must not be 'AtVertex' (not yet).\n";
      }
    } else if (algo_ == ByPropagatingMatched) {
      if (whichTrack1_ == None || whichState1_ == AtVertex) {
        algo_ = ByPropagatingMatchedTSCP;
        //throw cms::Exception("Configuration") << "Destination track must be non-null, and state must not be 'AtVertex' (not yet).\n";
      }
    } else if (algo_ == ByDirectComparison) {
      bool firstAtVertex = (whichTrack1_ == None || whichState1_ == AtVertex);
      bool secAtVertex = (whichTrack2_ == None || whichState2_ == AtVertex);
      if (firstAtVertex) {
        if (!secAtVertex)
          throw cms::Exception("Configuration")
              << "When using 'byDirectComparison' with 'src' at vertex (or None), 'matched' must be at vertex too.\n";
      } else {
        if (secAtVertex)
          throw cms::Exception("Configuration")
              << "When using 'byDirectComparison' with 'src' not at vertex, 'matched' can't be at vertex or None.\n";
        if (whichState1_ != whichState2_)
          throw cms::Exception("Configuration") << "You can't use 'byDirectComparison' with non-matching states.\n";
      }
    }

    useChi2_ = iConfig.existsAs<bool>("computeChi2") ? iConfig.getParameter<bool>("computeChi2") : false;
    if (useChi2_) {
      if (whichTrack1_ == None && whichTrack2_ == None)
        throw cms::Exception("Configuration") << "Can't compute chi2s if both tracks are set to 'none'.\n";
      maxChi2_ = iConfig.getParameter<double>("maxChi2");
      chi2DiagonalOnly_ = iConfig.getParameter<bool>("chi2DiagonalOnly");
      if (algo_ == ByPropagatingSrc || algo_ == ByPropagatingMatched) {
        chi2UseVertex_ = iConfig.getParameter<bool>("chi2UsePosition");
      } else {
        chi2UseVertex_ = iConfig.getParameter<bool>("chi2UseVertex");
        if (algo_ == ByDirectComparison) {
          std::string choice = iConfig.getParameter<std::string>("chi2MomentumForDxy");
          if (choice == "src")
            chi2FirstMomentum_ = true;
          else if (choice != "matched")
            throw cms::Exception("Configuration") << "chi2MomentumForDxy must be 'src' or 'matched'\n";
        }
      }
    } else
      maxChi2_ = 1;

    if (sortBy_ == Chi2 && !useChi2_)
      throw cms::Exception("Configuration") << "Can't sort by chi2s if 'computeChi2s' is not set to true.\n";
  }
}

void MatcherUsingTracksAlgorithm::getConf(const edm::ParameterSet &iConfig,
                                          const std::string &whatFor,
                                          WhichTrack &whichTrack,
                                          WhichState &whichState) {
  std::string s_whichTrack = iConfig.getParameter<std::string>(whatFor + "Track");
  if (s_whichTrack == "none") {
    whichTrack = None;
  } else if (s_whichTrack == "tracker") {
    whichTrack = TrackerTk;
  } else if (s_whichTrack == "muon") {
    whichTrack = MuonTk;
  } else if (s_whichTrack == "global") {
    whichTrack = GlobalTk;
  } else
    throw cms::Exception("Configuration") << "Parameter 'useTrack' must be 'none', 'tracker', 'muon', 'global'\n";
  if ((whichTrack != None) && (algo_ != ByTrackRef)) {
    std::string s_whichState = iConfig.getParameter<std::string>(whatFor + "State");
    if (s_whichState == "atVertex") {
      whichState = AtVertex;
    } else if (s_whichState == "innermost") {
      whichState = Innermost;
    } else if (s_whichState == "outermost") {
      whichState = Outermost;
    } else
      throw cms::Exception("Configuration") << "Parameter 'useState' must be 'atVertex', 'innermost', 'outermost'\n";
  }
}

/// Try to match one track to another one. Return true if succeeded.
/// For matches not by ref, it will update deltaR, deltaLocalPos and deltaPtRel if the match suceeded
bool MatcherUsingTracksAlgorithm::match(const reco::Candidate &c1,
                                        const reco::Candidate &c2,
                                        float &deltR,
                                        float &deltEta,
                                        float &deltPhi,
                                        float &deltaLocalPos,
                                        float &deltaPtRel,
                                        float &chi2) const {
  if (!(srcCut_(c1) && matchedCut_(c2)))
    return false;
  if (requireSameCharge_ && (c1.charge() != c2.charge()))
    return false;
  switch (algo_) {
    case ByTrackRef: {
      reco::TrackRef t1 = getTrack(c1, whichTrack1_);
      reco::TrackRef t2 = getTrack(c2, whichTrack2_);
      if (t1.isNonnull()) {
        if (t1 == t2)
          return true;
        if (t1.id() != t2.id()) {
          edm::LogWarning("MatcherUsingTracksAlgorithm")
              << "Trying to match by reference tracks coming from different collections.\n";
        }
      }
    }
      [[fallthrough]];
    case ByPropagatingSrc: {
      FreeTrajectoryState start = startingState(c1, whichTrack1_, whichState1_);
      TrajectoryStateOnSurface target = targetState(c2, whichTrack2_, whichState2_);
      return matchWithPropagation(start, target, deltR, deltEta, deltPhi, deltaLocalPos, deltaPtRel, chi2);
    }
    case ByPropagatingMatched: {
      FreeTrajectoryState start = startingState(c2, whichTrack2_, whichState2_);
      TrajectoryStateOnSurface target = targetState(c1, whichTrack1_, whichState1_);
      return matchWithPropagation(start, target, deltR, deltEta, deltPhi, deltaLocalPos, deltaPtRel, chi2);
    }
    case ByPropagatingSrcTSCP: {
      FreeTrajectoryState start = startingState(c1, whichTrack1_, whichState1_);
      FreeTrajectoryState target = startingState(c2, whichTrack2_, whichState2_);
      return matchWithPropagation(start, target, deltR, deltEta, deltPhi, deltaLocalPos, deltaPtRel, chi2);
    }
    case ByPropagatingMatchedTSCP: {
      FreeTrajectoryState start = startingState(c2, whichTrack2_, whichState2_);
      FreeTrajectoryState target = startingState(c1, whichTrack1_, whichState1_);
      return matchWithPropagation(start, target, deltR, deltEta, deltPhi, deltaLocalPos, deltaPtRel, chi2);
    }

    case ByDirectComparison: {
      FreeTrajectoryState start = startingState(c1, whichTrack1_, whichState1_);
      FreeTrajectoryState otherstart = startingState(c2, whichTrack2_, whichState2_);
      return matchByDirectComparison(start, otherstart, deltR, deltEta, deltPhi, deltaLocalPos, deltaPtRel, chi2);
    }
  }
  return false;
}

/// Find the best match to another candidate, and return its index in the vector
/// For matches not by ref, it will update deltaR, deltaLocalPos and deltaPtRel if the match suceeded
/// Returns -1 if the match fails
int MatcherUsingTracksAlgorithm::match(const reco::Candidate &c1,
                                       const edm::View<reco::Candidate> &c2s,
                                       float &deltR,
                                       float &deltEta,
                                       float &deltPhi,
                                       float &deltaLocalPos,
                                       float &deltaPtRel,
                                       float &chi2) const {
  if (!srcCut_(c1))
    return -1;

  // working and output variables
  FreeTrajectoryState start;
  TrajectoryStateOnSurface target;
  int match = -1;

  // pre-fetch some states if needed
  if (algo_ == ByPropagatingSrc || algo_ == ByPropagatingSrcTSCP || algo_ == ByPropagatingMatchedTSCP ||
      algo_ == ByDirectComparison) {
    start = startingState(c1, whichTrack1_, whichState1_);
  } else if (algo_ == ByPropagatingMatched)
    target = targetState(c1, whichTrack1_, whichState1_);

  // loop on the  collection
  edm::View<reco::Candidate>::const_iterator it, ed;
  int i;
  for (it = c2s.begin(), ed = c2s.end(), i = 0; it != ed; ++it, ++i) {
    if (!matchedCut_(*it))
      continue;
    if (requireSameCharge_ && (c1.charge() != it->charge()))
      continue;
    bool exit = false;
    switch (algo_) {
      case ByTrackRef: {
        reco::TrackRef t1 = getTrack(c1, whichTrack1_);
        reco::TrackRef t2 = getTrack(*it, whichTrack2_);
        if (t1.isNonnull()) {
          if (t1 == t2) {
            match = i;
            exit = true;
          }
          if (t1.id() != t2.id()) {
            edm::LogWarning("MatcherUsingTracksAlgorithm")
                << "Trying to match by reference tracks coming from different collections.\n";
          }
        }
      } break;
      case ByPropagatingSrc:
      case ByPropagatingMatched: {
        if (algo_ == ByPropagatingMatched)
          start = startingState(*it, whichTrack2_, whichState2_);
        else if (algo_ == ByPropagatingSrc)
          target = targetState(*it, whichTrack2_, whichState2_);
        if (matchWithPropagation(start, target, deltR, deltEta, deltPhi, deltaLocalPos, deltaPtRel, chi2)) {
          match = i;
        }
      } break;
      case ByDirectComparison: {
        FreeTrajectoryState otherstart = startingState(*it, whichTrack2_, whichState2_);
        if (matchByDirectComparison(start, otherstart, deltR, deltEta, deltPhi, deltaLocalPos, deltaPtRel, chi2)) {
          match = i;
        }
      } break;
      case ByPropagatingSrcTSCP: {
        FreeTrajectoryState otherstart = startingState(*it, whichTrack2_, whichState2_);
        if (matchWithPropagation(start, otherstart, deltR, deltEta, deltPhi, deltaLocalPos, deltaPtRel, chi2)) {
          match = i;
        }
      } break;
      case ByPropagatingMatchedTSCP: {
        FreeTrajectoryState otherstart = startingState(*it, whichTrack2_, whichState2_);
        if (matchWithPropagation(otherstart, start, deltR, deltEta, deltPhi, deltaLocalPos, deltaPtRel, chi2)) {
          match = i;
        }
      } break;
    }
    if (exit)
      break;
  }

  return match;
}

void MatcherUsingTracksAlgorithm::init(const edm::EventSetup &iSetup) {
  magfield_ = iSetup.getHandle(magfieldToken_);
  propagator_ = iSetup.getHandle(propagatorToken_);
  geometry_ = iSetup.getHandle(geometryToken_);
}

reco::TrackRef MatcherUsingTracksAlgorithm::getTrack(const reco::Candidate &reco, WhichTrack whichTrack) const {
  reco::TrackRef tk;
  const reco::RecoCandidate *rc = dynamic_cast<const reco::RecoCandidate *>(&reco);
  if (rc == nullptr)
    throw cms::Exception("Invalid Data") << "Input object is not a RecoCandidate.\n";
  switch (whichTrack) {
    case TrackerTk:
      tk = rc->track();
      break;
    case MuonTk:
      tk = rc->standAloneMuon();
      break;
    case GlobalTk:
      tk = rc->combinedMuon();
      break;
    default:
      break;  // just to make gcc happy
  }
  return tk;
}

FreeTrajectoryState MatcherUsingTracksAlgorithm::startingState(const reco::Candidate &reco,
                                                               WhichTrack whichTrack,
                                                               WhichState whichState) const {
  FreeTrajectoryState ret;
  if (whichTrack != None) {
    reco::TrackRef tk = getTrack(reco, whichTrack);
    if (tk.isNull()) {
      ret = FreeTrajectoryState();
    } else {
      switch (whichState) {
        case AtVertex:
          ret = trajectoryStateTransform::initialFreeState(*tk, magfield_.product());
          break;
        case Innermost:
          ret = trajectoryStateTransform::innerFreeState(*tk, magfield_.product());
          break;
        case Outermost:
          ret = trajectoryStateTransform::outerFreeState(*tk, magfield_.product());
          break;
      }
    }
  } else {
    ret = FreeTrajectoryState(GlobalPoint(reco.vx(), reco.vy(), reco.vz()),
                              GlobalVector(reco.px(), reco.py(), reco.pz()),
                              reco.charge(),
                              magfield_.product());
  }
  return ret;
}

TrajectoryStateOnSurface MatcherUsingTracksAlgorithm::targetState(const reco::Candidate &reco,
                                                                  WhichTrack whichTrack,
                                                                  WhichState whichState) const {
  TrajectoryStateOnSurface ret;
  reco::TrackRef tk = getTrack(reco, whichTrack);
  if (tk.isNonnull()) {
    switch (whichState) {
      case Innermost:
        ret = trajectoryStateTransform::innerStateOnSurface(*tk, *geometry_, magfield_.product());
        break;
      case Outermost:
        ret = trajectoryStateTransform::outerStateOnSurface(*tk, *geometry_, magfield_.product());
        break;
      default:
        break;  // just to make gcc happy
    }
  }
  return ret;
}

bool MatcherUsingTracksAlgorithm::matchWithPropagation(const FreeTrajectoryState &start,
                                                       const TrajectoryStateOnSurface &target,
                                                       float &lastDeltaR,
                                                       float &lastDeltaEta,
                                                       float &lastDeltaPhi,
                                                       float &lastDeltaLocalPos,
                                                       float &lastGlobalDPtRel,
                                                       float &lastChi2) const {
  if ((start.momentum().mag() == 0) || !target.isValid())
    return false;

  TrajectoryStateOnSurface tsos = propagator_->propagate(start, target.surface());

  bool isBest = false;
  if (tsos.isValid()) {
    float thisLocalPosDiff = (tsos.localPosition() - target.localPosition()).mag();
    float thisGlobalMomDeltaR = deltaR(tsos.globalMomentum(), target.globalMomentum());
    float thisGlobalMomDeltaPhi = fabs(deltaPhi(tsos.globalMomentum().barePhi(), target.globalMomentum().barePhi()));
    float thisGlobalMomDeltaEta = fabs(tsos.globalMomentum().eta() - target.globalMomentum().eta());
    float thisGlobalDPtRel =
        (tsos.globalMomentum().perp() - target.globalMomentum().perp()) / target.globalMomentum().perp();

    if ((thisLocalPosDiff < maxLocalPosDiff_) && (thisGlobalMomDeltaR < maxGlobalMomDeltaR_) &&
        (thisGlobalMomDeltaEta < maxGlobalMomDeltaEta_) && (thisGlobalMomDeltaPhi < maxGlobalMomDeltaPhi_) &&
        (fabs(thisGlobalDPtRel) < maxGlobalDPtRel_)) {
      float thisChi2 = useChi2_ ? getChi2(target, tsos, chi2DiagonalOnly_, chi2UseVertex_) : 0;
      if (thisChi2 >= maxChi2_)
        return false;
      switch (sortBy_) {
        case LocalPosDiff:
          isBest = (thisLocalPosDiff < lastDeltaLocalPos);
          break;
        case GlobalMomDeltaR:
          isBest = (thisGlobalMomDeltaR < lastDeltaR);
          break;
        case GlobalMomDeltaEta:
          isBest = (thisGlobalMomDeltaEta < lastDeltaEta);
          break;
        case GlobalMomDeltaPhi:
          isBest = (thisGlobalMomDeltaPhi < lastDeltaPhi);
          break;
        case GlobalDPtRel:
          isBest = (thisGlobalDPtRel < lastGlobalDPtRel);
          break;
        case Chi2:
          isBest = (thisChi2 < lastChi2);
          break;
      }
      if (isBest) {
        lastDeltaLocalPos = thisLocalPosDiff;
        lastDeltaR = thisGlobalMomDeltaR;
        lastDeltaEta = thisGlobalMomDeltaEta;
        lastDeltaPhi = thisGlobalMomDeltaPhi;
        lastGlobalDPtRel = thisGlobalDPtRel;
        lastChi2 = thisChi2;
      }
    }  // if match
  }

  return isBest;
}

bool MatcherUsingTracksAlgorithm::matchWithPropagation(const FreeTrajectoryState &start,
                                                       const FreeTrajectoryState &target,
                                                       float &lastDeltaR,
                                                       float &lastDeltaEta,
                                                       float &lastDeltaPhi,
                                                       float &lastDeltaLocalPos,
                                                       float &lastGlobalDPtRel,
                                                       float &lastChi2) const {
  if ((start.momentum().mag() == 0) || (target.momentum().mag() == 0))
    return false;
  TSCPBuilderNoMaterial propagator;
  /*2.2.X*/ try {
    TrajectoryStateClosestToPoint tscp = propagator(start, target.position());
    // if (!tscp.isValid()) return false;  // in 3.1.X

    bool isBest = false;
    float thisLocalPosDiff = (tscp.position() - target.position()).mag();
    float thisGlobalMomDeltaR = deltaR(tscp.momentum(), target.momentum());
    float thisGlobalMomDeltaPhi = fabs(deltaPhi(tscp.momentum().barePhi(), target.momentum().barePhi()));
    float thisGlobalMomDeltaEta = fabs(tscp.momentum().eta() - target.momentum().eta());
    float thisGlobalDPtRel = (tscp.momentum().perp() - target.momentum().perp()) / target.momentum().perp();

    if ((thisLocalPosDiff < maxLocalPosDiff_) && (thisGlobalMomDeltaR < maxGlobalMomDeltaR_) &&
        (thisGlobalMomDeltaEta < maxGlobalMomDeltaEta_) && (thisGlobalMomDeltaPhi < maxGlobalMomDeltaPhi_) &&
        (fabs(thisGlobalDPtRel) < maxGlobalDPtRel_)) {
      float thisChi2 = useChi2_ ? getChi2(target, tscp, chi2DiagonalOnly_, chi2UseVertex_) : 0;
      if (thisChi2 >= maxChi2_)
        return false;
      switch (sortBy_) {
        case LocalPosDiff:
          isBest = (thisLocalPosDiff < lastDeltaLocalPos);
          break;
        case GlobalMomDeltaR:
          isBest = (thisGlobalMomDeltaR < lastDeltaR);
          break;
        case GlobalMomDeltaEta:
          isBest = (thisGlobalMomDeltaEta < lastDeltaEta);
          break;
        case GlobalMomDeltaPhi:
          isBest = (thisGlobalMomDeltaPhi < lastDeltaPhi);
          break;
        case GlobalDPtRel:
          isBest = (thisGlobalDPtRel < lastGlobalDPtRel);
          break;
        case Chi2:
          isBest = (thisChi2 < lastChi2);
          break;
      }
      if (isBest) {
        lastDeltaLocalPos = thisLocalPosDiff;
        lastDeltaR = thisGlobalMomDeltaR;
        lastDeltaEta = thisGlobalMomDeltaEta;
        lastDeltaPhi = thisGlobalMomDeltaPhi;
        lastGlobalDPtRel = thisGlobalDPtRel;
        lastChi2 = thisChi2;
      }
    }  // if match

    return isBest;
    /*2.2.X*/ } catch (const TrajectoryStateException &err) { return false; }
}

bool MatcherUsingTracksAlgorithm::matchByDirectComparison(const FreeTrajectoryState &start,
                                                          const FreeTrajectoryState &target,
                                                          float &lastDeltaR,
                                                          float &lastDeltaEta,
                                                          float &lastDeltaPhi,
                                                          float &lastDeltaLocalPos,
                                                          float &lastGlobalDPtRel,
                                                          float &lastChi2) const {
    if ((start.momentum().mag() == 0) || target.momentum().mag() == 0)
      return false;

    bool isBest = false;
    float thisLocalPosDiff = (start.position() - target.position()).mag();
    float thisGlobalMomDeltaR = deltaR(start.momentum(), target.momentum());
    float thisGlobalMomDeltaPhi = fabs(deltaPhi(start.momentum().barePhi(), target.momentum().barePhi()));
    float thisGlobalMomDeltaEta = fabs(start.momentum().eta() - target.momentum().eta());
    float thisGlobalDPtRel = (start.momentum().perp() - target.momentum().perp()) / target.momentum().perp();

    if ((thisLocalPosDiff < maxLocalPosDiff_) && (thisGlobalMomDeltaR < maxGlobalMomDeltaR_) &&
        (thisGlobalMomDeltaEta < maxGlobalMomDeltaEta_) && (thisGlobalMomDeltaPhi < maxGlobalMomDeltaPhi_) &&
        (fabs(thisGlobalDPtRel) < maxGlobalDPtRel_)) {
      float thisChi2 = useChi2_ ? getChi2(start, target, chi2DiagonalOnly_, chi2UseVertex_, chi2FirstMomentum_) : 0;
      if (thisChi2 >= maxChi2_)
        return false;
      switch (sortBy_) {
        case LocalPosDiff:
          isBest = (thisLocalPosDiff < lastDeltaLocalPos);
          break;
        case GlobalMomDeltaR:
          isBest = (thisGlobalMomDeltaR < lastDeltaR);
          break;
        case GlobalMomDeltaEta:
          isBest = (thisGlobalMomDeltaEta < lastDeltaEta);
          break;
        case GlobalMomDeltaPhi:
          isBest = (thisGlobalMomDeltaPhi < lastDeltaPhi);
          break;
        case GlobalDPtRel:
          isBest = (thisGlobalDPtRel < lastGlobalDPtRel);
          break;
        case Chi2:
          isBest = (thisChi2 < lastChi2);
          break;
      }
      if (isBest) {
        lastDeltaLocalPos = thisLocalPosDiff;
        lastDeltaR = thisGlobalMomDeltaR;
        lastDeltaEta = thisGlobalMomDeltaEta;
        lastDeltaPhi = thisGlobalMomDeltaPhi;
        lastGlobalDPtRel = thisGlobalDPtRel;
        lastChi2 = thisChi2;
      }
    }  // if match

    return isBest;
}

double MatcherUsingTracksAlgorithm::getChi2(const FreeTrajectoryState &start,
                                            const FreeTrajectoryState &other,
                                            bool diagonalOnly,
                                            bool useVertex,
                                            bool useFirstMomentum) {
    if (!start.hasError() && !other.hasError())
      throw cms::Exception("LogicError") << "At least one of the two states must have errors to make chi2s.\n";
    AlgebraicSymMatrix55 cov;
    if (start.hasError())
      cov += start.curvilinearError().matrix();
    if (other.hasError())
      cov += other.curvilinearError().matrix();
    cropAndInvert(cov, diagonalOnly, !useVertex);
    GlobalVector p1 = start.momentum(), p2 = other.momentum();
    GlobalPoint x1 = start.position(), x2 = other.position();
    GlobalVector p = useFirstMomentum ? p1 : p2;
    double pt = p.perp(), pm = p.mag();
    double dsz =
        (x1.z() - x2.z()) * pt / pm - ((x1.x() - x2.x()) * p.x() + (x1.y() - x2.y()) * p.y()) / pt * p.z() / pm;
    double dxy = (-(x1.x() - x2.x()) * p.y() + (x1.y() - x2.y()) * p.x()) / pt;
    AlgebraicVector5 diff(start.charge() / p1.mag() - other.charge() / p2.mag(),
                          p1.theta() - p2.theta(),
                          (p1.phi() - p2.phi()).value(),
                          dxy,
                          dsz);
    return ROOT::Math::Similarity(diff, cov);
}

double MatcherUsingTracksAlgorithm::getChi2(const FreeTrajectoryState &start,
                                            const TrajectoryStateClosestToPoint &other,
                                            bool diagonalOnly,
                                            bool useVertex) {
    if (!start.hasError() && !other.hasError())
      throw cms::Exception("LogicError") << "At least one of the two states must have errors to make chi2s.\n";
    double pt;  // needed by pgconvert
    AlgebraicSymMatrix55 cov;
    if (start.hasError())
      cov += PerigeeConversions::ftsToPerigeeError(start).covarianceMatrix();
    if (other.hasError())
      cov += other.perigeeError().covarianceMatrix();
    cropAndInvert(cov, diagonalOnly, !useVertex);
    AlgebraicVector5 pgpar1 = PerigeeConversions::ftsToPerigeeParameters(start, other.referencePoint(), pt).vector();
    AlgebraicVector5 pgpar2 = other.perigeeParameters().vector();
    AlgebraicVector5 diff(pgpar1 - pgpar2);
    return ROOT::Math::Similarity(diff, cov);
}

double MatcherUsingTracksAlgorithm::getChi2(const TrajectoryStateOnSurface &start,
                                            const TrajectoryStateOnSurface &other,
                                            bool diagonalOnly,
                                            bool usePosition) {
    if (!start.hasError() && !other.hasError())
      throw cms::Exception("LogicError") << "At least one of the two states must have errors to make chi2s.\n";
    AlgebraicSymMatrix55 cov;
    if (start.hasError())
      cov += start.localError().matrix();
    if (other.hasError())
      cov += other.localError().matrix();
    cropAndInvert(cov, diagonalOnly, !usePosition);
    AlgebraicVector5 diff(start.localParameters().mixedFormatVector() - other.localParameters().mixedFormatVector());
    return ROOT::Math::Similarity(diff, cov);
}

void MatcherUsingTracksAlgorithm::cropAndInvert(AlgebraicSymMatrix55 &cov, bool diagonalOnly, bool top3by3only) {
    if (!top3by3only) {
      if (diagonalOnly) {
        for (size_t i = 0; i < 5; ++i) {
          for (size_t j = i + 1; j < 5; ++j) {
            cov(i, j) = 0;
          }
        }
      }
      cov.Invert();
    } else {
      // get 3x3 covariance
      AlgebraicSymMatrix33 momCov = cov.Sub<AlgebraicSymMatrix33>(0, 0);  // get 3x3 matrix
      if (diagonalOnly) {
        momCov(0, 1) = 0;
        momCov(0, 2) = 0;
        momCov(1, 2) = 0;
      }
      // invert
      momCov.Invert();
      // place it
      cov.Place_at(momCov, 0, 0);
      // zero the rest
      for (size_t i = 3; i < 5; ++i) {
        for (size_t j = i; j < 5; ++j) {
          cov(i, j) = 0;
        }
      }
    }
}
