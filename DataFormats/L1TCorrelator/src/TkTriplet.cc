// Original Author:   George Karathanasis,
//                    georgios.karathanasis@cern.ch, CU Boulder
// -*- C++ -*-
// Package:     L1Trigger
// Class  :     TkTriplet
// Description: Class to store the output of track-triplet producer, which used by L1T to create X->3h candidates (like W->3pi).

#include "DataFormats/L1TCorrelator/interface/TkTriplet.h"

using namespace l1t;

TkTriplet::TkTriplet()
    : charge_(-99.), pair_mass_max_(-99), pair_mass_min_(-99), pair_dz_max_(-99), pair_dz_min_(-99) {}

TkTriplet::TkTriplet(const LorentzVector& p4, int charge)
    : L1Candidate(p4), charge_(charge), pair_mass_max_(-99), pair_mass_min_(-99), pair_dz_max_(-99), pair_dz_min_(-99) {}

TkTriplet::TkTriplet(const LorentzVector& p4,
                     int charge,
                     double pair_mass_max,
                     double pair_mass_min,
                     double pair_dz_max,
                     double pair_dz_min,
                     std::vector<edm::Ptr<L1TTTrackType>> trkPtrList)
    : L1Candidate(p4),
      charge_(charge),
      pair_mass_max_(pair_mass_max),
      pair_mass_min_(pair_mass_min),
      pair_dz_max_(pair_dz_max),
      pair_dz_min_(pair_dz_min),
      trkPtrList_(trkPtrList) {}

int TkTriplet::bx() const {
  // in the producer TkJetProducer.cc, we keep only jets with bx = 0
  return 0;
}
