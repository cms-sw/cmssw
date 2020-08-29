#ifndef CommonTools_Utils_KinematicColumns_h
#define CommonTools_Utils_KinematicColumns_h

#include "FWCore/SOA/interface/Column.h"

namespace edm::soa {

  namespace col {

    SOA_DECLARE_COLUMN(Eta, float, "eta");
    SOA_DECLARE_COLUMN(Phi, float, "phi");
    SOA_DECLARE_COLUMN(Theta, float, "theta");

    SOA_DECLARE_COLUMN(Vz, float, "vz");

    SOA_DECLARE_COLUMN(Px, float, "px");
    SOA_DECLARE_COLUMN(Py, float, "py");
    SOA_DECLARE_COLUMN(Pz, float, "pz");
    SOA_DECLARE_COLUMN(Pt, float, "pt");
    SOA_DECLARE_COLUMN(P, float, "p");

  }  // namespace col

  SOA_DECLARE_DEFAULT(col::Eta, eta());
  SOA_DECLARE_DEFAULT(col::Phi, phi());
  SOA_DECLARE_DEFAULT(col::Theta, theta());
  SOA_DECLARE_DEFAULT(col::Vz, vz());
  SOA_DECLARE_DEFAULT(col::Px, px());
  SOA_DECLARE_DEFAULT(col::Py, py());
  SOA_DECLARE_DEFAULT(col::Pz, pz());
  SOA_DECLARE_DEFAULT(col::Pt, pt());
  SOA_DECLARE_DEFAULT(col::P, p());

}  // namespace edm::soa

#endif
