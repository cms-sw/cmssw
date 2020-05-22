#ifndef L1Trigger_TrackFindingTMTT_ConverterToTTTrack_h
#define L1Trigger_TrackFindingTMTT_ConverterToTTTrack_h

#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

//=== Convert non-persistent L1 track collection to the official persistent CMSSW EDM TTTrack format.
//=== Works for both L1track3D and for L1fittedTrk objects.

namespace tmtt {

  typedef edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> > TTStubDetSetVec;
  typedef edm::Ref<TTStubDetSetVec, TTStub<Ref_Phase2TrackerDigi_> > TTStubRef;

  class ConverterToTTTrack {
  public:
    // Initialize constants.
    ConverterToTTTrack(const Settings* settings) : settings_(settings), invPtToInvR_(settings->invPtToInvR()) {}

    // Convert L1fittedTrack or L1track3D (track candidates after/before fit) to TTTrack format.
    TTTrack<Ref_Phase2TrackerDigi_> makeTTTrack(const L1trackBase* trk,
                                                unsigned int iPhiSec,
                                                unsigned int iEtaReg) const;

  private:
    // Get references to stubs on track. (Works for either L1track3D or L1fittedTrack).
    std::vector<TTStubRef> stubRefs(const L1trackBase* trk) const;

  private:
    const Settings* settings_;  // Configuration parameters.
    float invPtToInvR_;         // converts 1/Pt to 1/radius_of_curvature
  };

}  // namespace tmtt
#endif
