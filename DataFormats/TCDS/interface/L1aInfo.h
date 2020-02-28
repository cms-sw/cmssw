#ifndef DATAFORMATS_TCDS_L1AINFO_H
#define DATAFORMATS_TCDS_L1AINFO_H

//---------------------------------------------------------------------------
//!  \class L1aInfo
//!  \brief Class to contain L1 accept history information from TCDS FED
//!
//!  \author Remi Mommsen - Fermilab
//---------------------------------------------------------------------------

#include <cstdint>
#include <ostream>

#include "DataFormats/TCDS/interface/TCDSRaw.h"

class L1aInfo {
public:
  L1aInfo();

  L1aInfo(const tcds::L1aInfo_v1&);

  // The history index, where -1 means the previous L1 accept, -2 the one before that, etc.
  int16_t getIndex() const { return index_; }

  // The orbit number when the L1 accept occured
  uint64_t getOrbitNr() const { return orbitNr_; }

  // The bunch-crossing counter for the L1 accept
  uint16_t getBXID() const { return bxid_; }

  // The event type of the L1 accept corresponding to edm::EventAuxiliary::ExperimentType
  uint8_t getEventType() const { return eventType_; }

private:
  uint64_t orbitNr_;
  uint16_t bxid_;
  int16_t index_;
  uint8_t eventType_;
};

/// Pretty-print operator for L1aInfo
std::ostream& operator<<(std::ostream&, const L1aInfo&);

#endif  // DATAFORMATS_TCDS_L1AINFO_H
