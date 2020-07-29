#ifndef CommonTools_Utils_KinematicColumns_h
#define CommonTools_Utils_KinematicColumns_h

#include "FWCore/SOA/interface/Column.h"

namespace edm {

  namespace soa {

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

  }  // namespace soa

}  // namespace edm

#endif
