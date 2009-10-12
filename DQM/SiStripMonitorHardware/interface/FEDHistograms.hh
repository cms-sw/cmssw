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

#include "DQM/SiStripMonitorHardware/interface/HistogramBase.hh"
#include "DQM/SiStripMonitorHardware/interface/FEDErrors.hh"

class FEDHistograms: public HistogramBase {

public:
  
  FEDHistograms();

  ~FEDHistograms();
  
  //initialise histograms
  void initialise(const edm::ParameterSet& iConfig,
		  std::ostringstream* pDebugStream
		  );

  void fillCountersHistograms(const FEDErrors::FEDCounters & aFedLevelCounters, 
			      const FEDErrors::ChannelCounters & aChLevelCounters, 
			      const double aTime);

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



   //book the top level histograms
  void bookTopLevelHistograms(DQMStore* dqm);

  //book individual FED histograms or book all FED level histograms at once
  void bookFEDHistograms(unsigned int fedId,
			 bool fullDebugMode = false
			 );

  void bookAllFEDHistograms();

  std::string tkHistoMapName(unsigned int aIndex=0);

  TkHistoMap * tkHistoMapPointer(unsigned int aIndex=0);

protected:
  
private:

  //counting histograms (histogram of number of problems per event)
  MonitorElement *nFEDErrors_, 
    *nFEDDAQProblems_, 
    *nFEDsWithFEProblems_, 
    *nFEDCorruptBuffers_, 
    *nBadChannelStatusBits_,
    *nBadActiveChannelStatusBits_,
    *nUnconnectedChannels_,
    *nFEDsWithFEOverflows_, 
    *nFEDsWithFEBadMajorityAddresses_, 
    *nFEDsWithMissingFEs_;

  MonitorElement *nAPVStatusBit_;
  MonitorElement *nAPVError_;
  MonitorElement *nAPVAddressError_;
  MonitorElement *nUnlocked_;
  MonitorElement *nOutOfSync_;

  MonitorElement *nTotalBadChannelsvsTime_;
  MonitorElement *nTotalBadActiveChannelsvsTime_;

  MonitorElement *nFEDErrorsvsTime_;
  MonitorElement *nFEDCorruptBuffersvsTime_;
  MonitorElement *nFEDsWithFEProblemsvsTime_;

  MonitorElement *nAPVStatusBitvsTime_;
  MonitorElement *nAPVErrorvsTime_;
  MonitorElement *nAPVAddressErrorvsTime_;
  MonitorElement *nUnlockedvsTime_;
  MonitorElement *nOutOfSyncvsTime_;

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
