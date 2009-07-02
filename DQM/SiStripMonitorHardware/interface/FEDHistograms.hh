// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      FEDHistograms
// 
/**\class FEDHistograms DQM/SiStripMonitorHardware/interface/FEDHistograms.hh

 Description: DQM source application to produce data integrety histograms for SiStrip data
*/
//
// Original Author:  Nicholas Cripps in plugin file
//         Created:  2008/09/16
// Modified by    :  Anne-Marie Magnan, code copied from plugin to this class
//

#ifndef DQM_SiStripMonitorHardware_FEDHistograms_HH
#define DQM_SiStripMonitorHardware_FEDHistograms_HH

#include <sstream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripMonitorHardware/interface/FEDErrors.hh"

class FEDHistograms {

public:
  
  struct HistogramConfig {
    bool enabled;
    unsigned int nBins;
    double min;
    double max;
  };
  

  FEDHistograms();

  ~FEDHistograms();
  
  //initialise histograms
  void initialise(const edm::ParameterSet& iConfig,
		  std::ostringstream* pDebugStream
		  );

  //fill a histogram if the pointer is not NULL (ie if it has been booked)
  void fillHistogram(MonitorElement* histogram, 
		     double value,
		     double weight=1.
		     );

  void fillCountersHistograms(const FEDErrors::FEDCounters & aFedLevelCounters, const unsigned int aEvtNum);

  void fillFEDHistograms(FEDErrors & aFedError, 
			 bool lFullDebug
			 );

  void fillFEHistograms(const unsigned int aFedId,
			const FEDErrors::FELevelErrors & aFeLevelErrors
			);

  void fillChannelsHistograms(const unsigned int aFedId, 
			      const FEDErrors::ChannelLevelErrors & aChErr, 
			      bool fullDebug
			      );

  void fillAPVsHistograms(const unsigned int aFedId, 
			  const FEDErrors::APVLevelErrors & aAPVErr, 
			  bool fullDebug
			  );



  //fill tkHistoMap of percentage of bad channels per module
  void fillTkHistoMap(uint32_t & detid,
		      float value
		      );
 
  //book the top level histograms
  void bookTopLevelHistograms(DQMStore* dqm);

  //book individual FED histograms or book all FED level histograms at once
  void bookFEDHistograms(unsigned int fedId,
			 bool fullDebugMode = false
			 );

  bool isTkHistoMapEnabled();

  void bookAllFEDHistograms();

  //load the config for a histogram from PSet called <configName>HistogramConfig (writes a debug message to stream if pointer is non-NULL)
  void getConfigForHistogram(const std::string& configName, 
			     const edm::ParameterSet& psetContainingConfigPSet,
			     std::ostringstream* pDebugStream
			     );

  //book an individual hiostogram if enabled in config
  MonitorElement* bookHistogram(const std::string& configName,
				const std::string& name, 
				const std::string& title,
                                const unsigned int nBins, 
				const double min, 
				const double max,
                                const std::string& xAxisTitle
				);

  //same but using binning from config
  MonitorElement* bookHistogram(const std::string& configName,
				const std::string& name, 
				const std::string& title, 
				const std::string& xAxisTitle
				);
  
protected:
  
private:

  DQMStore* dqm_;

  //config for histograms (enabled? bins)
  std::map<std::string,HistogramConfig> histogramConfig_;

  //counting histograms (histogram of number of problems per event)
  MonitorElement *nFEDErrors_, 
    *nFEDDAQProblems_, 
    *nFEDsWithFEProblems_, 
    *nFEDCorruptBuffers_, 
    *nBadActiveChannelStatusBits_,
    *nFEDsWithFEOverflows_, 
    *nFEDsWithFEBadMajorityAddresses_, 
    *nFEDsWithMissingFEs_;

  MonitorElement *nBadChannelsvsEvtNum_;

  //top level histograms
  MonitorElement *anyFEDErrors_, 
    *anyDAQProblems_, 
    *corruptBuffers_, 
    *invalidBuffers_, 
    *badIDs_, 
    *badChannelStatusBits_, 
    *badActiveChannelStatusBits_,
    *badDAQCRCs_, 
    *badFEDCRCs_, 
    *badDAQPacket_, 
    *dataMissing_, 
    *dataPresent_, 
    *feOverflows_, 
    *badMajorityAddresses_, 
    *feMissing_, 
    *anyFEProblems_;

  //FED level histograms
  std::map<unsigned int,MonitorElement*> feOverflowDetailed_, 
    badMajorityAddressDetailed_, 
    feMissingDetailed_;

  std::map<unsigned int,MonitorElement*> badStatusBitsDetailed_, 
    apvErrorDetailed_, 
    apvAddressErrorDetailed_, 
    unlockedDetailed_, 
    outOfSyncDetailed_;

  //has individual FED histogram been booked? (index is FedId)
  std::vector<bool> histosBooked_, 
    debugHistosBooked_;

  std::string tkMapConfigName_;
  TkHistoMap *tkmapFED_;



};//class



#endif //DQM_SiStripMonitorHardware_FEDHistograms_HH
