#ifndef MuonAnalysis_MuonAssociators_interface_PropagateToMuonSetup_h
#define MuonAnalysis_MuonAssociators_interface_PropagateToMuonSetup_h
//
//

/**
  \class    PropagateToMuonSetup PropagateToMuonSetup.h "HLTriggerOffline/Muon/interface/PropagateToMuonSetup.h" 

  \brief Propagate an object (usually a track) to the second (default) or first muon station.

*/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuon.h"
#include "MuonAnalysis/MuonAssociators/interface/trackStateEnums.h"

class IdealMagneticFieldRecord;
class TrackingComponentsRecord;
class MuonRecoGeometryRecord;

class PropagateToMuonSetup {
public:
  explicit PropagateToMuonSetup(const edm::ParameterSet &iConfig, edm::ConsumesCollector);
  ~PropagateToMuonSetup(){};

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

#endif
