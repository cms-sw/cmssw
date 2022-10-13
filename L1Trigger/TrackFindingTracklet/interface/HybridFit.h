#ifndef L1Trigger_TrackFindingTracklet_interface_HybridFit_h
#define L1Trigger_TrackFindingTracklet_interface_HybridFit_h

#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"

#ifdef USEHYBRID
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFTrackletTrack.h"
#endif

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

namespace trklet {

  class Stub;
  class L1TStub;
  class Tracklet;

  class HybridFit {
  public:
    HybridFit(unsigned int iSector, Settings const& settings, Globals* globals);

    ~HybridFit() = default;

    void Fit(Tracklet* tracklet, std::vector<const Stub*>& trackstublist);

  private:
    unsigned int iSector_;

    Settings const& settings_;
    Globals* globals_;
  };
};  // namespace trklet
#endif
