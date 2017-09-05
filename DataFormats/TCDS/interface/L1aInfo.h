#ifndef DATAFORMATS_TCDS_L1AINFO_H
#define DATAFORMATS_TCDS_L1AINFO_H

//---------------------------------------------------------------------------
//!  \class L1aInfo
//!  \brief Class to contain L1 accept history information from TCDS FED
//!
//!  \author Remi Mommsen - Fermilab
//---------------------------------------------------------------------------

#include <stdint.h>
#include <ostream>

#include "DataFormats/TCDS/interface/TCDSRaw.h"


class L1aInfo
{

public:

  L1aInfo();

  L1aInfo(const tcds::L1aInfo_v1&);

  int16_t getIndex() const { return index_; }
  uint64_t getOrbitNr() const { return orbitNr_; }
  uint16_t getBXID() const { return bxid_; }
  uint8_t getEventType() const { return eventType_; }

private:

  int16_t index_;
  uint64_t orbitNr_;
  uint16_t bxid_;
  uint8_t eventType_;

};

/// Pretty-print operator for L1aInfo
std::ostream& operator<<(std::ostream&, const L1aInfo&);

#endif // DATAFORMATS_TCDS_L1AINFO_H
