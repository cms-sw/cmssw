#ifndef DataFormatsL1TCorrelator_TkElectron_h
#define DataFormatsL1TCorrelator_TkElectron_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkEm
//

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>

namespace l1t {

  class TkElectron : public TkEm {
  public:
    typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
    typedef std::vector<L1TTTrackType> L1TTTrackCollection;

    TkElectron();

    TkElectron(const LorentzVector& p4,
               const edm::Ref<EGammaBxCollection>& egRef,
               const edm::Ptr<L1TTTrackType>& trkPtr,
               float tkisol = -999.);

    // ---------- const member functions ---------------------

    const edm::Ptr<L1TTTrackType>& trkPtr() const { return trkPtr_; }

    float trkzVtx() const { return trkzVtx_; }
    double trackCurvature() const { return trackCurvature_; }
    float compositeBdtScore() const { return compositeBdtScore_; }
    // ---------- member functions ---------------------------

    void setTrkzVtx(float TrkzVtx) { trkzVtx_ = TrkzVtx; }
    void setTrackCurvature(double trackCurvature) { trackCurvature_ = trackCurvature; }
    void setCompositeBdtScore(float score) { compositeBdtScore_ = score; }


  private:
    edm::Ptr<L1TTTrackType> trkPtr_;
    float trkzVtx_;
    double trackCurvature_;
    float compositeBdtScore_;
  };
}  // namespace l1t
#endif
