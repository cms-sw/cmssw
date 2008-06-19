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

class CastorUnpacker {
public:

  struct Collections {
    Collections();
    std::vector<CastorDataFrame>* castorCont;
    std::vector<HcalCalibDataFrame>* calibCont;
    std::vector<HcalTriggerPrimitiveDigi>* tpCont;
  };

  /// for normal data
  CastorUnpacker(int sourceIdOffset, int beg, int end) ;
  //CastorUnpacker(int sourceIdOffset, int beg, int end) : sourceIdOffset_(sourceIdOffset), startSample_(beg), endSample_(end) { }
  /// For histograms, no begin and end
  // CastorUnpacker(int sourceIdOffset) : sourceIdOffset_(sourceIdOffset), startSample_(-1), endSample_(-1) { }
  // void unpack(const FEDRawData& raw, const CastorElectronicsMap& emap, std::vector<HcalHistogramDigi>& histoDigis);
  void unpack(const FEDRawData& raw, const CastorElectronicsMap& emap, Collections& conts, HcalUnpackerReport& report);
  // Old -- deprecated
  // void unpack(const FEDRawData& raw, const CastorElectronicsMap& emap, std::vector<CastorDataFrame>& precision, std::vector<HcalTriggerPrimitiveDigi>& tp);
private:
  int sourceIdOffset_; ///< number to subtract from the source id to get the dcc id
  int startSample_; ///< first sample from fed raw data to copy 
  int endSample_; ///< last sample from fed raw data to copy (if present)
  std::set<CastorElectronicsId> unknownIds_,unknownIdsTrig_; ///< Recorded to limit number of times a log message is generated
};

#endif // CastorUnpacker_h_included
