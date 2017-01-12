/* -*- C++ -*- */
#ifndef HcalUnpacker_h_included
#define HcalUnpacker_h_included 1

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalCalibDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/HOTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/HcalHistogramDigi.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "DataFormats/HcalDigi/interface/HcalTTPDigi.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalUMNioDigi.h"
#include <set>

class HcalUnpacker {
public:

  struct Collections {
    Collections();
    std::vector<HBHEDataFrame>* hbheCont;
    std::vector<HODataFrame>* hoCont;
    std::vector<HFDataFrame>* hfCont;
    std::vector<HcalCalibDataFrame>* calibCont;
    std::vector<ZDCDataFrame>* zdcCont;
    std::vector<HcalTriggerPrimitiveDigi>* tpCont;
    std::vector<HOTriggerPrimitiveDigi>* tphoCont;
    std::vector<HcalTTPDigi>* ttp;
    QIE10DigiCollection* qie10;
    QIE11DigiCollection* qie11;

  };

  /// for normal data
  HcalUnpacker(int sourceIdOffset, int beg, int end) : sourceIdOffset_(sourceIdOffset), startSample_(beg), endSample_(end), expectedOrbitMessageTime_(-1), mode_(0) { }
  /// For histograms, no begin and end
  HcalUnpacker(int sourceIdOffset) : sourceIdOffset_(sourceIdOffset), startSample_(-1), endSample_(-1),  expectedOrbitMessageTime_(-1), mode_(0) { }
  void setExpectedOrbitMessageTime(int time) { expectedOrbitMessageTime_=time; }
  void unpack(const FEDRawData& raw, const HcalElectronicsMap& emap, std::vector<HcalHistogramDigi>& histoDigis);
  void unpack(const FEDRawData& raw, const HcalElectronicsMap& emap, Collections& conts, HcalUnpackerReport& report, bool silent=false);
  void setMode(int mode) { mode_=mode; }
private:
  void unpackVME(const FEDRawData& raw, const HcalElectronicsMap& emap, Collections& conts, HcalUnpackerReport& report, bool silent=false);
  void unpackUTCA(const FEDRawData& raw, const HcalElectronicsMap& emap, Collections& conts, HcalUnpackerReport& report, bool silent=false);
  void unpackUMNio(const FEDRawData& raw, int slot, HcalUMNioDigi& umnio);


  int sourceIdOffset_; ///< number to subtract from the source id to get the dcc id
  int startSample_; ///< first sample from fed raw data to copy 
  int endSample_; ///< last sample from fed raw data to copy (if present)
  int expectedOrbitMessageTime_; ///< Expected orbit bunch time (needed to evaluate time differences)
  int mode_;
  std::set<HcalElectronicsId> unknownIds_,unknownIdsTrig_; ///< Recorded to limit number of times a log message is generated
};

#endif // HcalUnpacker_h_included
