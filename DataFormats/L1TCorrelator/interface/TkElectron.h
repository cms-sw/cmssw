#ifndef DataFormatsL1TCorrelator_TkElectron_h
#define DataFormatsL1TCorrelator_TkElectron_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkEm
//

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
               const edm::Ptr<L1Candidate>& egCaloPtr,
               const edm::Ptr<L1TTTrackType>& trkPtr,
               float tkisol = -999.);

    TkElectron(const LorentzVector& p4, float tkisol = -999.);

    // ---------- const member functions ---------------------

    const edm::Ptr<L1TTTrackType>& trkPtr() const { return trkPtr_; }

    float trkzVtx() const { return trkzVtx_; }
    float idScore() const { return idScore_; }
    // ---------- member functions ---------------------------

    void setTrkPtr(const edm::Ptr<L1TTTrackType>& tkPtr) { trkPtr_ = tkPtr; }
    void setTrkzVtx(float TrkzVtx) { trkzVtx_ = TrkzVtx; }
    void setIdScore(float score) { idScore_ = score; }

    l1gt::Electron hwObj() const {
      if (encoding() != HWEncoding::GT) {
        throw cms::Exception("RuntimeError") << "TkElectron::hwObj : encoding is not in GT format!" << std::endl;
      }
      return l1gt::Electron::unpack_ap(egBinaryWord<l1gt::Electron::BITWIDTH>());
    }

  private:
    edm::Ptr<L1TTTrackType> trkPtr_;
    float trkzVtx_;
    float idScore_;
  };
}  // namespace l1t
#endif
