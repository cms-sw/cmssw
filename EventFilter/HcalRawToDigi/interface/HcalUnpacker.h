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
  void unpack(const raw::FEDRawData& raw, const cms::hcal::HcalMapping& emap, std::vector<cms::HBHEDataFrame>& precision, std::vector<cms::HcalTriggerPrimitiveDigi>& tp);
  void unpack(const raw::FEDRawData& raw, const cms::hcal::HcalMapping& emap, std::vector<cms::HODataFrame>& precision, std::vector<cms::HcalTriggerPrimitiveDigi>& tp);
  void unpack(const raw::FEDRawData& raw, const cms::hcal::HcalMapping& emap, std::vector<cms::HFDataFrame>& precision, std::vector<cms::HcalTriggerPrimitiveDigi>& tp);
private:
  int sourceIdOffset_; ///< number to subtract from the source id to get the dcc id
  int startSample_; ///< first sample from fed raw data to copy
  int endSample_; ///< last sample from fed raw data to copy (if present)
};

#endif // HcalUnpacker_h_included
