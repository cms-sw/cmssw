/* -*- C++ -*- */
#ifndef CastorUnpacker_h_included
#define CastorUnpacker_h_included 1

#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalCalibDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
// #include "DataFormats/HcalDigi/interface/HcalHistogramDigi.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include <set>
#include "EventFilter/CastorRawToDigi/interface/CastorRawCollections.h"

class CastorUnpacker {
public:

  /// for normal data
  CastorUnpacker(int sourceIdOffset, int beg, int end) ;
  void setExpectedOrbitMessageTime(int time) { expectedOrbitMessageTime_=time; }
  /// For histograms, no begin and end
  // void unpack(const FEDRawData& raw, const CastorElectronicsMap& emap, std::vector<HcalHistogramDigi>& histoDigis);
  void unpack(const FEDRawData& raw, const CastorElectronicsMap& emap, CastorRawCollections& conts, HcalUnpackerReport& report, bool silent=false);
private:
  int sourceIdOffset_; ///< number to subtract from the source id to get the dcc id
  int startSample_; ///< first sample from fed raw data to copy 
  int endSample_; ///< last sample from fed raw data to copy (if present)
  int expectedOrbitMessageTime_; ///< Expected orbit bunch time (needed to evaluate time differences)
  std::set<CastorElectronicsId> unknownIds_,unknownIdsTrig_; ///< Recorded to limit number of times a log message is generated
};

#endif // CastorUnpacker_h_included
