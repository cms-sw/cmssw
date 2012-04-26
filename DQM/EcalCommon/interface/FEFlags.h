#ifndef EcalDQMFEFlags_H
#define EcalDQMFEFlags_H

namespace ecaldqm {

  // partially taken from EventFilter/EcalRawToDigi/interface/DCCRawDataDefinitions.h
  enum FEFlags {
    Enabled = 0,
    Disabled = 1,
    Timeout = 2,
    HeaderError = 3,
    ChannelId = 4,
    LinkError = 5,
    BlockSize = 6,
    Suppressed = 7,
    FIFOFull = 8,
    L1ADesync = 9,
    BXDesync = 10,
    L1ABXDesync = 11,
    FIFOFullL1ADesync = 12,
    HParity = 13,
    VParity = 14,
    ForcedZS = 15,
    nFEFlags = 16
  };

}

#endif
