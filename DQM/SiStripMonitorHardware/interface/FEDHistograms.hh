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

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class FEDHistograms {

public:
  
  struct HistogramConfig {
    bool enabled;
    unsigned int nBins;
    double min;
    double max;
  };
  
  struct FEDCounters {
    unsigned int nFEDErrors;
    unsigned int nDAQProblems;
    unsigned int nFEDsWithFEProblems;
    unsigned int nCorruptBuffers;
    unsigned int nBadActiveChannels;
    unsigned int nFEDsWithFEOverflows;
    unsigned int nFEDsWithFEBadMajorityAddresses;
    unsigned int nFEDsWithMissingFEs;
  };

  struct FECounters {
    unsigned int nFEOverflows; 
    unsigned int nFEBadMajorityAddresses; 
    unsigned int nFEMissing;
  };


  FEDHistograms();

  ~FEDHistograms();
  
  //initialise histograms
  void initialise(const edm::ParameterSet& iConfig,
		  std::ostringstream* pDebugStream
		  );

  void fillCountersHistogram(const FEDCounters & fedLevelCounters );


  //fill a histogram if the pointer is not NULL (ie if it has been booked)
  void fillHistogram(const std::string & histoName, 
		     const double value,
		     const int fedId = -1
		     );

  //book the top level histograms
  void bookTopLevelHistograms(DQMStore* dqm,
			      const std::string & folderName);

  //book individual FED histograms or book all FED level histograms at once
  void bookFEDHistograms(unsigned int fedId,
			 bool fullDebugMode = false
			 );

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
  std::string dqmPath_;

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






};//class



#endif //DQM_SiStripMonitorHardware_FEDHistograms_HH
