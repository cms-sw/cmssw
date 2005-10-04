/* -*- C++ -*- */
#ifndef HcalUnpacker_h_included
#define HcalUnpacker_h_included 1

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "CondFormats/HcalMapping/interface/HcalMapping.h"

class HcalUnpacker {
public:
  HcalUnpacker(int sourceIdOffset, int beg=0, int end=9) : sourceIdOffset_(sourceIdOffset), startSample_(beg), endSample_(end) { }
  void unpack(const FEDRawData& raw, const HcalMapping& emap, std::vector<HBHEDataFrame>& precision, std::vector<HcalTriggerPrimitiveDigi>& tp);
  void unpack(const FEDRawData& raw, const HcalMapping& emap, std::vector<HODataFrame>& precision, std::vector<HcalTriggerPrimitiveDigi>& tp);
  void unpack(const FEDRawData& raw, const HcalMapping& emap, std::vector<HFDataFrame>& precision, std::vector<HcalTriggerPrimitiveDigi>& tp);
private:
  int sourceIdOffset_; ///< number to subtract from the source id to get the dcc id
  int startSample_; ///< first sample from fed raw data to copy
  int endSample_; ///< last sample from fed raw data to copy (if present)
};

#endif // HcalUnpacker_h_included
