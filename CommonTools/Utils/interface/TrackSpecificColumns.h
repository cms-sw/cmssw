#ifndef CommonTools_Utils_TrackSpecificColumns_h
#define CommonTools_Utils_TrackSpecificColumns_h

#include "FWCore/SOA/interface/Column.h"

namespace edm::soa {

  namespace col {

    SOA_DECLARE_COLUMN(PtError, float, "ptError");
    SOA_DECLARE_COLUMN(D0, float, "d0");
    SOA_DECLARE_COLUMN(NumberOfValidHits, int, "numberOfValidHits");
    SOA_DECLARE_COLUMN(MissingInnerHits, int, "missingInnerHits");
    SOA_DECLARE_COLUMN(Charge, int, "charge");

  }  // namespace col

  SOA_DECLARE_DEFAULT(col::PtError, ptError());
  SOA_DECLARE_DEFAULT(col::D0, d0());
  SOA_DECLARE_DEFAULT(col::NumberOfValidHits, numberOfValidHits());
  SOA_DECLARE_DEFAULT(col::Charge, charge());
  SOA_DECLARE_DEFAULT(col::MissingInnerHits, missingInnerHits());

}  // namespace edm::soa

#endif
