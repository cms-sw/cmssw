#ifndef DQM_SiStripMonitorHardware_SPYHistograms_HH
#define DQM_SiStripMonitorHardware_SPYHistograms_HH

#include <sstream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripMonitorHardware/interface/SiStripSpyUtilities.h"
#include "DQM/SiStripMonitorHardware/interface/HistogramBase.hh"

class SPYHistograms: public HistogramBase {

 public:

  struct Trends {
    unsigned int meanDigitalLow; // digitalLow averaged over all channels
  };

  struct ErrorCounters {
    unsigned int nNoData; //max=min=0
    unsigned int nLowRange; // max-min < min value to tune
    unsigned int nHighRange; // max-min > max value to tune
    unsigned int nMinZero; // min = 0
    unsigned int nMaxSat; //max = 1023
    unsigned int nLowPb; //min < min value to tune but > 0
    unsigned int nHighPb; //max > max value to tune but < 1023
    unsigned int nOOS;//header+trailer found with right distance between them but not in expected position
                      // or header found above 16 (no trailer found)
                      // or 2*2-high separated by 70 samples found (last in right position...)
    unsigned int nOtherPbs;
    unsigned int nAPVError;//number of APVs with error bit 0
    unsigned int nAPVAddressError;//number of APV pairs with different APV addresses
    unsigned int nNegPeds;//ped subtr value = 0
  };

  //helper structs to fill histograms
  struct Errors {
    bool hasNoData;
    bool hasLowRange; 
    bool hasHighRange;
    bool hasMinZero;
    bool hasMaxSat;
    bool hasLowPb;
    bool hasHighPb;
    bool hasOOS;
    bool hasOtherPbs;
    bool hasErrorBit0;
    bool hasErrorBit1;
    bool hasAPVAddressError0;
    bool hasAPVAddressError1;
    bool hasNegPeds;
  };

  SPYHistograms();
  
  ~SPYHistograms() override;
  
  //initialise histograms
  void initialise(const edm::ParameterSet& iConfig,
		  std::ostringstream* pDebugStream
		  ) override;

  //book the top level histograms
  void bookTopLevelHistograms(DQMStore::IBooker &);

  //book individual FED histograms or book all FED level histograms at once
  void bookFEDHistograms(DQMStore::IBooker & , const unsigned int fedId,
			 const Errors & aErr,
			 bool doAll = false);

  void bookAllFEDHistograms(DQMStore::IBooker &);

  void fillCountersHistograms(const ErrorCounters & aCounter, const double aTime);

  void fillGainHistograms(const Trends & aTrendElement, const double aTime);

  void fillFEDHistograms(const Errors & aErr, const unsigned int aFedId);

  void fillDetailedHistograms(const Errors & aErr,
			      const sistrip::SpyUtilities::Frame & aFrame,
			      const unsigned int aFedId, 
			      const unsigned int aFedChannel);

  bool tkHistoMapEnabled(unsigned int aIndex=0) override{
    return false;
  };

  TkHistoMap * tkHistoMapPointer(unsigned int aIndex=0) override{
    return nullptr;
};

 protected:
  
 private:
 //histos

  //counting histograms (histogram of number of problems per event)
  HistogramConfig nNoData_;
  HistogramConfig nLowRange_;
  HistogramConfig nHighRange_;
  HistogramConfig nMinZero_;
  HistogramConfig nMaxSat_;
  HistogramConfig nLowPb_;
  HistogramConfig nHighPb_;
  HistogramConfig nOutOfSync_;
  HistogramConfig nOtherPbs_;
  HistogramConfig nApvErrorBit_;
  HistogramConfig nApvAddressError_;
  HistogramConfig nNegativePeds_;

  //vsTime
  HistogramConfig nNoDatavsTime_;
  HistogramConfig nLowRangevsTime_;
  HistogramConfig nHighRangevsTime_;
  HistogramConfig nMinZerovsTime_;
  HistogramConfig nMaxSatvsTime_;
  HistogramConfig nLowPbvsTime_;
  HistogramConfig nHighPbvsTime_;
  HistogramConfig nOutOfSyncvsTime_;
  HistogramConfig nOtherPbsvsTime_;
  HistogramConfig nApvErrorBitvsTime_;
  HistogramConfig nApvAddressErrorvsTime_;
  HistogramConfig nNegativePedsvsTime_;
  HistogramConfig meanDigitalLowvsTime_;

 //top level histograms
  HistogramConfig noData_;
  HistogramConfig lowRange_;
  HistogramConfig highRange_;
  HistogramConfig minZero_;
  HistogramConfig maxSat_;
  HistogramConfig lowPb_;
  HistogramConfig highPb_;
  HistogramConfig outOfSync_;
  HistogramConfig otherPbs_;
  HistogramConfig apvErrorBit_;
  HistogramConfig apvAddressError_;
  HistogramConfig negativePeds_;

  HistogramConfig frameRange_;
  HistogramConfig frameMin_;
  HistogramConfig frameMax_;
  HistogramConfig baseline_;


  //FED level histograms
  HistogramConfig  noDataDetailed_;
  HistogramConfig  lowRangeDetailed_;
  HistogramConfig  highRangeDetailed_;
  HistogramConfig  minZeroDetailed_;
  HistogramConfig  maxSatDetailed_;
  HistogramConfig  lowPbDetailed_;
  HistogramConfig  highPbDetailed_;
  HistogramConfig  outOfSyncDetailed_;
  HistogramConfig  otherPbsDetailed_;
  HistogramConfig  apvErrorBitDetailed_;
  HistogramConfig  apvAddressErrorDetailed_;
  HistogramConfig  negativePedsDetailed_;

  HistogramConfig  positionOfFirstHeaderBitDetailed_;
  HistogramConfig  positionOfFirstTrailerBitDetailed_;
  HistogramConfig  distanceHeaderTrailerDetailed_;

  std::map<unsigned int,MonitorElement* > noDataDetailedMap_;
  std::map<unsigned int,MonitorElement* > lowRangeDetailedMap_;
  std::map<unsigned int,MonitorElement* > highRangeDetailedMap_;
  std::map<unsigned int,MonitorElement* > minZeroDetailedMap_;
  std::map<unsigned int,MonitorElement* > maxSatDetailedMap_;
  std::map<unsigned int,MonitorElement* > lowPbDetailedMap_;
  std::map<unsigned int,MonitorElement* > highPbDetailedMap_;
  std::map<unsigned int,MonitorElement* > outOfSyncDetailedMap_;
  std::map<unsigned int,MonitorElement* > otherPbsDetailedMap_;
  std::map<unsigned int,MonitorElement* > apvErrorBitDetailedMap_;
  std::map<unsigned int,MonitorElement* > apvAddressErrorDetailedMap_;
  std::map<unsigned int,MonitorElement* > negativePedsDetailedMap_;

  std::map<unsigned int,MonitorElement* > positionOfFirstHeaderBitDetailedMap_;
  std::map<unsigned int,MonitorElement* > positionOfFirstTrailerBitDetailedMap_;
  std::map<unsigned int,MonitorElement* > distanceHeaderTrailerDetailedMap_;

  std::vector<bool> histosBooked_;

};//class



#endif //DQM_SiStripMonitorHardware_SPYHistograms_HH


