#ifndef MuonAnalysis_MuonAssociators_interface_PropagateToMuonSetup_h
#define MuonAnalysis_MuonAssociators_interface_PropagateToMuonSetup_h

/**
  \class PropagateToMuonSetup PropagateToMuonSetup.h "MuonAnalysis/MuonAssociators/interface/PropagateToMuonSetup.h" 

  \brief Propagate an object (usually a track) to the second (default) or first muon station.
*/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuon.h"
#include "MuonAnalysis/MuonAssociators/interface/trackStateEnums.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

template <edm::Transition Tr>
class PropagateToMuonSetupT {
public:
  explicit PropagateToMuonSetupT(const edm::ParameterSet &iConfig, edm::ConsumesCollector);
  ~PropagateToMuonSetupT() = default;

  /// Call this method at the beginning of each run, to initialize geometry,
  /// magnetic field and propagators
  PropagateToMuon init(const edm::EventSetup &iSetup) const;
  static void fillPSetDescription(edm::ParameterSetDescription &desc) {
    desc.add<bool>("useSimpleGeometry", true);
    desc.add<bool>("useStation2", true);
    desc.add<bool>("fallbackToME1", false);
    desc.add<bool>("cosmicPropagationHypothesis", false);
    desc.add<bool>("useMB2InOverlap", false);
    desc.add<std::string>("useTrack", "tracker");
    desc.add<std::string>("useState", "atVertex");
    desc.add<edm::ESInputTag>("propagatorAlong", edm::ESInputTag("", "hltESPSteppingHelixPropagatorAlong"));
    desc.add<edm::ESInputTag>("propagatorAny", edm::ESInputTag("", "SteppingHelixPropagatorAny"));
    desc.add<edm::ESInputTag>("propagatorOpposite", edm::ESInputTag("", "hltESPSteppingHelixPropagatorOpposite"));
  }

private:
  /// Use simplified geometry (cylinders and disks, not individual chambers)
  const bool useSimpleGeometry_;

  /// Propagate to MB2 (default) instead of MB1
  const bool useMB2_;

  /// Fallback to ME1 if propagation to ME2 fails
  const bool fallbackToME1_;

  /// Labels for input collections
  WhichTrack whichTrack_;
  WhichState whichState_;

  /// for cosmics, some things change: the along-opposite is not in-out, nor the innermost/outermost states are in-out really
  const bool cosmicPropagation_;

  const bool useMB2InOverlap_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_, propagatorAnyToken_,
      propagatorOppositeToken_;
  const edm::ESGetToken<MuonDetLayerGeometry, MuonRecoGeometryRecord> muonGeometryToken_;
};

template <edm::Transition Tr>
PropagateToMuonSetupT<Tr>::PropagateToMuonSetupT(const edm::ParameterSet &iConfig, edm::ConsumesCollector iC)
    : useSimpleGeometry_(iConfig.getParameter<bool>("useSimpleGeometry")),
      useMB2_(iConfig.getParameter<bool>("useStation2")),
      fallbackToME1_(iConfig.getParameter<bool>("fallbackToME1")),
      whichTrack_(None),
      whichState_(AtVertex),
      cosmicPropagation_(iConfig.getParameter<bool>("cosmicPropagationHypothesis")),
      useMB2InOverlap_(iConfig.getParameter<bool>("useMB2InOverlap")),
      magfieldToken_(iC.esConsumes<Tr>()),
      propagatorToken_(iC.esConsumes<Tr>(iConfig.getParameter<edm::ESInputTag>("propagatorAlong"))),
      propagatorAnyToken_(iC.esConsumes<Tr>(iConfig.getParameter<edm::ESInputTag>("propagatorAny"))),
      propagatorOppositeToken_(iC.esConsumes<Tr>(iConfig.getParameter<edm::ESInputTag>("propagatorOpposite"))),
      muonGeometryToken_(iC.esConsumes<Tr>()) {
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

template <edm::Transition Tr>
PropagateToMuon PropagateToMuonSetupT<Tr>::init(const edm::EventSetup &iSetup) const {
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

using PropagateToMuonSetup = PropagateToMuonSetupT<edm::Transition::Event>;

#endif
