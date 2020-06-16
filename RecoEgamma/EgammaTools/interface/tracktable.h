#ifndef RecoEgamma_EgammaTools_tracktable_h
#define RecoEgamma_EgammaTools_tracktable_h

#include "FWCore/SOA/interface/Column.h"
#include "FWCore/SOA/interface/Table.h"

namespace egamma {

namespace tracktable {
  SOA_DECLARE_COLUMN(Pt, double, "pt");
  SOA_DECLARE_COLUMN(Eta, double, "eta");
  SOA_DECLARE_COLUMN(Phi, double, "phi");
  SOA_DECLARE_COLUMN(Vz, double, "vz");

  using EtaPhiTable = edm::soa::Table<Eta, Phi>;
  using SimpleTrackTable = edm::soa::Table<Pt, Eta, Phi, Vz>;
}

}

#endif
