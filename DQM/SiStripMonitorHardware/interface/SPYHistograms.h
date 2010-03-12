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
  
  ~SPYHistograms();
  
  //initialise histograms
  void initialise(const edm::ParameterSet& iConfig,
		  std::ostringstream* pDebugStream
		  );

  //book the top level histograms
  void bookTopLevelHistograms(DQMStore* dqm);

  //book individual FED histograms or book all FED level histograms at once
  void bookFEDHistograms(const unsigned int fedId,
			 const Errors & aErr,
			 bool doAll = false);

  void bookAllFEDHistograms();

  void fillCountersHistograms(const ErrorCounters & aCounter, const double aTime);

  void fillGainHistograms(const Trends & aTrendElement, const double aTime);

  void fillFEDHistograms(const Errors & aErr, const unsigned int aFedId);

  void fillDetailedHistograms(const Errors & aErr,
			      const sistrip::SpyUtilities::Frame & aFrame,
			      const unsigned int aFedId, 
			      const unsigned int aFedChannel);

  std::string tkHistoMapName(unsigned int aIndex=0){
    return "";
  };

  TkHistoMap * tkHistoMapPointer(unsigned int aIndex=0){
    return 0;
};

 protected:
  
 private:
 //histos

  //counting histograms (histogram of number of problems per event)
  MonitorElement *nNoData_;
  MonitorElement *nLowRange_;
  MonitorElement *nHighRange_;
  MonitorElement *nMinZero_;
  MonitorElement *nMaxSat_;
  MonitorElement *nLowPb_;
  MonitorElement *nHighPb_;
  MonitorElement *nOutOfSync_;
  MonitorElement *nOtherPbs_;
  MonitorElement *nApvErrorBit_;
  MonitorElement *nApvAddressError_;
  MonitorElement *nNegativePeds_;

  //vsTime
  MonitorElement *nNoDatavsTime_;
  MonitorElement *nLowRangevsTime_;
  MonitorElement *nHighRangevsTime_;
  MonitorElement *nMinZerovsTime_;
  MonitorElement *nMaxSatvsTime_;
  MonitorElement *nLowPbvsTime_;
  MonitorElement *nHighPbvsTime_;
  MonitorElement *nOutOfSyncvsTime_;
  MonitorElement *nOtherPbsvsTime_;
  MonitorElement *nApvErrorBitvsTime_;
  MonitorElement *nApvAddressErrorvsTime_;
  MonitorElement *nNegativePedsvsTime_;
  MonitorElement *meanDigitalLowvsTime_;

 //top level histograms
  MonitorElement *noData_;
  MonitorElement *lowRange_;
  MonitorElement *highRange_;
  MonitorElement *minZero_;
  MonitorElement *maxSat_;
  MonitorElement *lowPb_;
  MonitorElement *highPb_;
  MonitorElement *outOfSync_;
  MonitorElement *otherPbs_;
  MonitorElement *apvErrorBit_;
  MonitorElement *apvAddressError_;
  MonitorElement *negativePeds_;

  MonitorElement *frameRange_;
  MonitorElement *frameMin_;
  MonitorElement *frameMax_;
  MonitorElement *baseline_;


  //FED level histograms
  std::map<unsigned int,MonitorElement *> noDataDetailed_;
  std::map<unsigned int,MonitorElement *> lowRangeDetailed_;
  std::map<unsigned int,MonitorElement *> highRangeDetailed_;
  std::map<unsigned int,MonitorElement *> minZeroDetailed_;
  std::map<unsigned int,MonitorElement *> maxSatDetailed_;
  std::map<unsigned int,MonitorElement *> lowPbDetailed_;
  std::map<unsigned int,MonitorElement *> highPbDetailed_;
  std::map<unsigned int,MonitorElement *> outOfSyncDetailed_;
  std::map<unsigned int,MonitorElement *> otherPbsDetailed_;
  std::map<unsigned int,MonitorElement *> apvErrorBitDetailed_;
  std::map<unsigned int,MonitorElement *> apvAddressErrorDetailed_;
  std::map<unsigned int,MonitorElement *> negativePedsDetailed_;

  std::map<unsigned int,MonitorElement *> positionOfFirstHeaderBitDetailed_;
  std::map<unsigned int,MonitorElement *> positionOfFirstTrailerBitDetailed_;
  std::map<unsigned int,MonitorElement *> distanceHeaderTrailerDetailed_;

  std::vector<bool> histosBooked_;

};//class



#endif //DQM_SiStripMonitorHardware_SPYHistograms_HH


