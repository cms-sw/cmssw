#ifndef CSCChamberTimeCorrections_h
#define CSCChamberTimeCorrections_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>
#include <string>

class CSCChamberTimeCorrections{
 public:
  CSCChamberTimeCorrections();
  ~CSCChamberTimeCorrections();
  
  struct ChamberTimeCorrections{
    short int cfeb_length;
    char cfeb_rev;
    short int alct_length;
    char alct_rev;
    short int cfeb_tmb_skew_delay;
    short int cfeb_timing_corr;
    short int cfeb_cable_delay;
    short int anode_bx_offset;
  };
  int factor_precision;

  enum factors{FCORR=100};

  // accessor to appropriate element ->should be chamber
  const ChamberTimeCorrections & item(const CSCDetId & cscId) const;
  
  typedef std::vector<ChamberTimeCorrections> ChamberContainer;

  ChamberContainer chamberCorrections;
};

#endif
