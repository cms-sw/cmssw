#ifndef L1Trigger_CSCBitWidths_h
#define L1Trigger_CSCBitWidths_h

/**
 * \class CSCBitWidths
 * \remark Collection of stuff from ORCA
 *
 * Static interface to hold bit widths of various CSC data.
 */

class CSCBitWidths
{
 public:
  enum clct_bits { CLCT_PATTERN_BITS = 4 };

  enum corrlct_bits { kPatternBitWidth=CLCT_PATTERN_BITS, kQualityBitWidth = 4, kBendBitWidth = 1 };

  enum addresses { kLocalPhiAddressWidth = 19, kGlobalEtaAddressWidth = kLocalPhiAddressWidth };

  enum data_sizes { kLocalPhiDataBitWidth = 10, kLocalPhiBendDataBitWidth = 6 , kGlobalEtaBitWidth = 7};
};

#endif
