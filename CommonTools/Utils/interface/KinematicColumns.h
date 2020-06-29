#ifndef CommonTools_Utils_KinematicColumns_h
#define CommonTools_Utils_KinematicColumns_h

#include "FWCore/SOA/interface/Column.h"

namespace edm {

  namespace soa {

    namespace col {

      SOA_DECLARE_COLUMN(Eta, float, "eta");
      SOA_DECLARE_COLUMN(Phi, float, "phi");

      SOA_DECLARE_COLUMN(Vz, float, "vz");

      SOA_DECLARE_COLUMN(Px, float, "px");
      SOA_DECLARE_COLUMN(Py, float, "py");
      SOA_DECLARE_COLUMN(Pz, float, "pz");
      SOA_DECLARE_COLUMN(Pt, float, "pt");
      SOA_DECLARE_COLUMN(P, float, "p");

      SOA_DECLARE_COLUMN(PtError, float, "ptError");
      SOA_DECLARE_COLUMN(D0, float, "d0");

      SOA_DECLARE_COLUMN(NumberOfValidHits, int, "numberOfValidHits");
      SOA_DECLARE_COLUMN(MissingInnerHits, int, "missingInnerHits");
      SOA_DECLARE_COLUMN(Charge, int, "charge");

    }  // namespace col

  }  // namespace soa

}  // namespace edm

#endif
