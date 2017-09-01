#ifndef DATAFORMATS_TCDS_L1AINFO_H
#define DATAFORMATS_TCDS_L1AINFO_H

//---------------------------------------------------------------------------
//!  \class L1aInfo
//!  \brief Class to contain L1 accept history information from TCDS FED
//!
//!  \author Remi Mommsen - Fermilab
//---------------------------------------------------------------------------


#include "DataFormats/TCDS/interface/TCDSRaw.h"


class L1aInfo
{

public:

  L1aInfo();

  L1aInfo(const tcds::L1aInfo_v1&);

  uint64_t getorbit() const { return orbit_; }
  uint16_t getbxid() const { return bxid_; }
  unsigned char getEventType() const { return eventType_; }

private:

  uint64_t orbit_;
  uint16_t bxid_;
  unsigned char eventType_;

};

#endif // DATAFORMATS_TCDS_L1AINFO_H
