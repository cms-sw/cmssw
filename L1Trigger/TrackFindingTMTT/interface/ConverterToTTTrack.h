#ifndef __ConverterToTTTrack_H__
#define __ConverterToTTTrack_H__

#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrk4and5.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

using namespace std;

namespace TMTT {

typedef edmNew::DetSetVector< TTStub<Ref_Phase2TrackerDigi_> > DetSetVec;
typedef edm::Ref<DetSetVec, TTStub<Ref_Phase2TrackerDigi_> > TTStubRef;

//=== Convert our non-persistent L1 track collection to the official persistent CMSSW EDM TTTrack format.
//=== Works for both L1track3D and for L1fittedTrk4and5 objects.

class ConverterToTTTrack {

public:

  // Initialize constants.
  ConverterToTTTrack(const Settings* settings) : settings_(settings) {invPtToInvR_ = settings->invPtToInvR();}

  ~ConverterToTTTrack(){}

  // N.B. The function with argument L1fittedTrk4and5 below should be used if both 4 and 5 parameter helix fit results 
  // are to be stored in the same TTTrack object. Whilst the function with argument L1fittedTrack should be used if
  // they are two be stored in two different TTTrack objects.
  // One of these is a better idea, but we don't yet know which, so keep both functions!
  // N.B. Do not call these two functions for invalid fitted tracks.

  // Convert L1track3D (track candidate before fit) to TTTrack format.
  TTTrack< Ref_Phase2TrackerDigi_ > makeTTTrack(const L1track3D&        trk     , unsigned int iPhiSec, unsigned int iEtaReg) const;
  // Convert L1fittedTrack (track candidate after fit) to TTTrack format.
  TTTrack< Ref_Phase2TrackerDigi_ > makeTTTrack(const L1fittedTrack&    trk     , unsigned int iPhiSec, unsigned int iEtaReg) const;
  // Convert L1fittedTrk4and5 (track candidate after fit) to TTTrack format.
  TTTrack< Ref_Phase2TrackerDigi_ > makeTTTrack(const L1fittedTrk4and5& trk4and5, unsigned int iPhiSec, unsigned int iEtaReg) const;

private:

  // Get references to stubs on track. (Templated, so works for either L1track3D or L1fittedTrack).
  template<class T> 
  std::vector<TTStubRef> getStubRefs(const T& trk) const {

    std::vector<TTStubRef> ttstubrefs;
    const std::vector<const Stub*> stubs = trk.getStubs();
    for (size_t ii = 0; ii < stubs.size(); ii++) {
	TTStubRef ref = *stubs.at(ii);
	ttstubrefs.push_back(ref);
    }

    return ttstubrefs;
  }

private:

  const Settings *settings_; // Configuration parameters.
  float invPtToInvR_; // converts 1/Pt to 1/radius_of_curvature
};

}
#endif
