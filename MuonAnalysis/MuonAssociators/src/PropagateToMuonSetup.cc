#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuonSetup.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

PropagateToMuonSetup::PropagateToMuonSetup(const edm::ParameterSet &iConfig, edm::ConsumesCollector iC)
    : useSimpleGeometry_(iConfig.getParameter<bool>("useSimpleGeometry")),
      useMB2_(iConfig.getParameter<bool>("useStation2")),
      fallbackToME1_(iConfig.getParameter<bool>("fallbackToME1")),
      whichTrack_(None),
      whichState_(AtVertex),
      cosmicPropagation_(iConfig.getParameter<bool>("cosmicPropagationHypothesis")),
      useMB2InOverlap_(iConfig.getParameter<bool>("useMB2InOverlap")),
      magfieldToken_(iC.esConsumes()),
      propagatorToken_(iC.esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorAlong"))),
      propagatorAnyToken_(iC.esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorAny"))),
      propagatorOppositeToken_(iC.esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorOpposite"))),
      muonGeometryToken_(iC.esConsumes()) {
  std::string whichTrack = iConfig.getParameter<std::string>("useTrack");
  if (whichTrack == "none") {
    whichTrack_ = None;
  } else if (whichTrack == "tracker") {
    whichTrack_ = TrackerTk;
  } else if (whichTrack == "muon") {
    whichTrack_ = MuonTk;
  } else if (whichTrack == "global") {
    whichTrack_ = GlobalTk;
  } else
    throw cms::Exception("Configuration") << "Parameter 'useTrack' must be 'none', 'tracker', 'muon', 'global'\n";
  if (whichTrack_ != None) {
    std::string whichState = iConfig.getParameter<std::string>("useState");
    if (whichState == "atVertex") {
      whichState_ = AtVertex;
    } else if (whichState == "innermost") {
      whichState_ = Innermost;
    } else if (whichState == "outermost") {
      whichState_ = Outermost;
    } else
      throw cms::Exception("Configuration") << "Parameter 'useState' must be 'atVertex', 'innermost', "
                                               "'outermost'\n";
  }
  if (cosmicPropagation_ && (whichTrack_ == None || whichState_ == AtVertex)) {
    throw cms::Exception("Configuration") << "When using 'cosmicPropagationHypothesis' useTrack must not be "
                                             "'none', and the state must not be 'atVertex'\n";
  }
}

PropagateToMuon PropagateToMuonSetup::init(const edm::EventSetup &iSetup) const {
  auto const magfield = iSetup.getHandle(magfieldToken_);
  auto const propagator = iSetup.getHandle(propagatorToken_);
  auto const propagatorOpposite = iSetup.getHandle(propagatorOppositeToken_);
  auto const propagatorAny = iSetup.getHandle(propagatorAnyToken_);
  auto const muonGeometry = iSetup.getHandle(muonGeometryToken_);

  return PropagateToMuon(magfield,
                         propagator,
                         propagatorAny,
                         propagatorOpposite,
                         muonGeometry,
                         useSimpleGeometry_,
                         useMB2_,
                         fallbackToME1_,
                         whichTrack_,
                         whichState_,
                         cosmicPropagation_,
                         useMB2InOverlap_);
}
