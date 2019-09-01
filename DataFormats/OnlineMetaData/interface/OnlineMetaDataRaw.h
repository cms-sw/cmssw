#ifndef DATAFORMATS_ONLINEMETADATA_ONLINEMETADATARAW_H
#define DATAFORMATS_ONLINEMETADATA_ONLINEMETADATARAW_H

//---------------------------------------------------------------------------
//!  \class OnlineMetaDataRaw
//!  \brief Structure of raw data from soft FED 1022
//!
//!  \author Remi Mommsen - Fermilab
//---------------------------------------------------------------------------

#include <cstdint>

namespace online {

  struct Luminosity_v1 {
    const uint64_t timestamp;
    const uint16_t lumiSection;
    const uint16_t lumiNibble;
    const float instLumi;
    const float avgPileUp;
  };

  struct BeamSpot_v1 {
    const uint64_t timestamp;
    const float x;
    const float y;
    const float z;
    const float dxdz;
    const float dydz;
    const float errX;
    const float errY;
    const float errZ;
    const float errDxdz;
    const float errDydz;
    const float widthX;
    const float widthY;
    const float sigmaZ;
    const float errWidthX;
    const float errWidthY;
    const float errSigmaZ;
  };

  struct CTPPS_v1 {
    uint64_t timestamp;
    uint64_t status;
  };

  struct DCS_v1 {
    const uint64_t timestamp;
    const uint32_t highVoltageReady;
    const float magnetCurrent;
  };

  struct Data_v1 {
    const uint8_t version;
    const Luminosity_v1 luminosity;
    const BeamSpot_v1 beamSpot;
    const DCS_v1 dcs;
  };

  struct Data_v2 {
    const uint8_t version;
    const Luminosity_v1 luminosity;
    const BeamSpot_v1 beamSpot;
    const CTPPS_v1 ctpps;
    const DCS_v1 dcs;
  };

}  // namespace online

#endif  // DATAFORMATS_ONLINEMETADATA_ONLINEMETADATARAW_H
