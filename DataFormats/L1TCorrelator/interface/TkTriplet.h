#ifndef DataFormatsL1TCorrelator_TkTriplet_h
#define DataFormatsL1TCorrelator_TkTriplet_h

// Original author: G Karathanasis,
//                    georgios.karathanasis@cern.ch, CU Boulder
// -*- C++ -*-
// Package:     L1Trigger
// Class  :     TkTriplet
// Description: Class to store the output of track-triplet producer, which used by L1T to create X->3h candidates (like W->3pi).

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include <vector>

namespace l1t {

  class TkTriplet : public L1Candidate {
  public:
    typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
    typedef std::vector<L1TTTrackType> L1TTTrackCollection;

    TkTriplet();

    TkTriplet(const LorentzVector& p4, int charge);
    TkTriplet(const LorentzVector& p4,
              int charge,
              double pair_mass_max,
              double pair_mass_min,
              double pair_dz_max,
              double pair_dz_min,
              std::vector<edm::Ptr<L1TTTrackType>> trkPtrList);

    ~TkTriplet() override{};

    int getTripletCharge() const { return charge_; }
    double getPairMassMax() const { return pair_mass_max_; }
    double getPairMassMin() const { return pair_mass_min_; }
    double getPairDzMax() const { return pair_dz_max_; }
    double getPairDzMin() const { return pair_dz_min_; }
    const edm::Ptr<L1TTTrackType>& trkPtr(size_t i) const { return trkPtrList_.at(i); }
    int bx() const;

  private:
    int charge_;
    double pair_mass_max_;
    double pair_mass_min_;
    double pair_dz_max_;
    double pair_dz_min_;
    std::vector<edm::Ptr<L1TTTrackType>> trkPtrList_;
  };
}  // namespace l1t

#endif
