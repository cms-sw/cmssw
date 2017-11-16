#include "DataFormats/GEMDigi/interface/GEMGEBStatusDigi.h"

GEMGEBStatusDigi::GEMGEBStatusDigi(uint32_t ZeroSup,
				   uint8_t InputID,
				   uint16_t Vwh,
				   uint16_t ErrorC,
				   uint16_t OHCRC,
				   uint16_t Vwt,
				   uint8_t InFu,
				   uint8_t Stuckd,
				   std::vector<uint8_t> v_GEBflags) :
  ZeroSup_(ZeroSup),
  InputID_(InputID),
  Vwh_(Vwh),
  ErrorC_(ErrorC),
  OHCRC_(OHCRC),
  Vwt_(Vwt),
  InFu_(InFu),
  Stuckd_(Stuckd),
  v_GEBflags_(v_GEBflags)
{};
