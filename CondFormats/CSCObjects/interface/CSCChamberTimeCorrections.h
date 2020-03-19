#ifndef CSCChamberTimeCorrections_h
#define CSCChamberTimeCorrections_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <iosfwd>
#include <vector>

class CSCChamberTimeCorrections {
public:
  CSCChamberTimeCorrections() {}
  ~CSCChamberTimeCorrections() {}

  struct ChamberTimeCorrections {
    short int cfeb_length;
    char cfeb_rev;
    short int alct_length;
    char alct_rev;
    short int cfeb_tmb_skew_delay;
    short int cfeb_timing_corr;
    short int cfeb_cable_delay;
    short int anode_bx_offset;

    COND_SERIALIZABLE;
  };
  int factor_precision;

  enum factors { FCORR = 100 };

  typedef std::vector<ChamberTimeCorrections> ChamberContainer;
  ChamberContainer chamberCorrections;

  const ChamberTimeCorrections& item(int index) const { return chamberCorrections[index]; }
  int precision() const { return factor_precision; }

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream& os, const CSCChamberTimeCorrections& cscdb);

#endif
