/* -*- C++ -*- */
#ifndef EventFilter_CastorRawToDigi_ZdcUnpacker_h
#define EventFilter_CastorRawToDigi_ZdcUnpacker_h 1

#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalCalibDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include <set>
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "EventFilter/CastorRawToDigi/interface/CastorRawCollections.h"


class ZdcUnpacker {
public:

  /// for normal data
  ZdcUnpacker(int sourceIdOffset, int beg, int end) ;
  void setExpectedOrbitMessageTime(int time) { expectedOrbitMessageTime_=time; }
  /// For histograms, no begin and end
  // void unpack(const FEDRawData& raw, const ZdcElectronicsMap& emap, std::vector<HcalHistogramDigi>& histoDigis);
  void unpack(const FEDRawData& raw, const CastorElectronicsMap& emap, CastorRawCollections& conts, HcalUnpackerReport& report, bool silent=false);
  void unpackOld(const FEDRawData& raw, const HcalElectronicsMap& emap, CastorRawCollections& conts, HcalUnpackerReport& report, bool silent=false);
private:
  int sourceIdOffset_; ///< number to subtract from the source id to get the dcc id
  int startSample_; ///< first sample from fed raw data to copy 
  int endSample_; ///< last sample from fed raw data to copy (if present)
  int expectedOrbitMessageTime_; ///< Expected orbit bunch time (needed to evaluate time differences)
  std::set<HcalElectronicsId> unknownIds_,unknownIdsTrig_; ///< Recorded to limit number of times a log message is generated
  int mode_;
};

#endif // ZdcUnpacker_h_included
