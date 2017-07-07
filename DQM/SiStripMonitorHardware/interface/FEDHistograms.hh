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

  ~FEDHistograms() override;
  
  //initialise histograms
  void initialise(const edm::ParameterSet& iConfig,
		  std::ostringstream* pDebugStream
		  ) override;

  void fillCountersHistograms(const FEDErrors::FEDCounters & aFedLevelCounters, 
			      const FEDErrors::ChannelCounters & aChLevelCounters,
			      const unsigned int aMaxSize,
			      const double aTime);

  void fillFEDHistograms(FEDErrors & aFedError,
			 const unsigned int aEvtSize,
			 bool lFullDebug
			 );

  void fillFEHistograms(const unsigned int aFedId,
			const FEDErrors::FELevelErrors & aFeLevelErrors,
			const FEDErrors::EventProperties & aEventProp 
			);

  void fillChannelsHistograms(const unsigned int aFedId, 
			      const FEDErrors::ChannelLevelErrors & aChErr, 
			      bool fullDebug
			      );

  void fillAPVsHistograms(const unsigned int aFedId, 
			  const FEDErrors::APVLevelErrors & aAPVErr, 
			  bool fullDebug
			  );

  void fillMajorityHistograms(const unsigned int aPart,
			      const float aValue,
			      const std::vector<unsigned int> & aFedIdVec);

  bool feMajHistosEnabled();

  void fillLumiHistograms(const FEDErrors::LumiErrors & aLumErr);

  bool cmHistosEnabled();

   //book the top level histograms
  void bookTopLevelHistograms(DQMStore::IBooker & , std::string topFolderName = "SiStrip");

  //book individual FED histograms or book all FED level histograms at once
  void bookFEDHistograms(DQMStore::IBooker & , unsigned int fedId,
			 bool fullDebugMode = false
			 );

  void bookAllFEDHistograms(DQMStore::IBooker & , bool);

  bool tkHistoMapEnabled(unsigned int aIndex=0) override;

  TkHistoMap * tkHistoMapPointer(unsigned int aIndex=0) override;

  MonitorElement *cmHistPointer(bool aApv1);

  MonitorElement *getFedvsAPVpointer();

protected:
  
private:

  //counting histograms (histogram of number of problems per event)

  HistogramConfig fedEventSize_;
  HistogramConfig fedMaxEventSizevsTime_;

  HistogramConfig nFEDErrors_, 
    nFEDDAQProblems_, 
    nFEDsWithFEProblems_, 
    nFEDCorruptBuffers_, 
    nBadChannelStatusBits_,
    nBadActiveChannelStatusBits_,
    nUnconnectedChannels_,
    nFEDsWithFEOverflows_, 
    nFEDsWithFEBadMajorityAddresses_, 
    nFEDsWithMissingFEs_;

  HistogramConfig nFEDErrorsvsTime_;
  HistogramConfig nFEDCorruptBuffersvsTime_;
  HistogramConfig nFEDsWithFEProblemsvsTime_;

  HistogramConfig nTotalBadChannels_;
  HistogramConfig nTotalBadActiveChannels_;

  HistogramConfig nTotalBadChannelsvsTime_;
  HistogramConfig nTotalBadActiveChannelsvsTime_;

  HistogramConfig nAPVStatusBit_;
  HistogramConfig nAPVError_;
  HistogramConfig nAPVAddressError_;
  HistogramConfig nUnlocked_;
  HistogramConfig nOutOfSync_;

  HistogramConfig nAPVStatusBitvsTime_;
  HistogramConfig nAPVErrorvsTime_;
  HistogramConfig nAPVAddressErrorvsTime_;
  HistogramConfig nUnlockedvsTime_;
  HistogramConfig nOutOfSyncvsTime_;

  //top level histograms
  HistogramConfig anyFEDErrors_, 
    anyDAQProblems_, 
    corruptBuffers_, 
    invalidBuffers_, 
    badIDs_, 
    badChannelStatusBits_, 
    badActiveChannelStatusBits_,
    badDAQCRCs_, 
    badFEDCRCs_, 
    badDAQPacket_, 
    dataMissing_, 
    dataPresent_, 
    feOverflows_, 
    badMajorityAddresses_,
    badMajorityInPartition_,
    feMissing_, 
    anyFEProblems_,
    fedIdVsApvId_;

  HistogramConfig feTimeDiffTIB_,
    feTimeDiffTOB_,
    feTimeDiffTECB_,
    feTimeDiffTECF_;

  HistogramConfig feTimeDiffvsDBX_;

  HistogramConfig apveAddress_;
  HistogramConfig feMajAddress_;

  HistogramConfig feMajFracTIB_;
  HistogramConfig feMajFracTOB_;
  HistogramConfig feMajFracTECB_;
  HistogramConfig feMajFracTECF_;

  HistogramConfig medianAPV0_;
  HistogramConfig medianAPV1_;

  HistogramConfig feOverflowDetailed_, 
    badMajorityAddressDetailed_, 
    feMissingDetailed_;

  HistogramConfig badStatusBitsDetailed_, 
    apvErrorDetailed_, 
    apvAddressErrorDetailed_, 
    unlockedDetailed_, 
    outOfSyncDetailed_;
  
  //FED level histograms
  std::map<unsigned int,MonitorElement*> feOverflowDetailedMap_, 
    badMajorityAddressDetailedMap_, 
    feMissingDetailedMap_;

  std::map<unsigned int,MonitorElement*> badStatusBitsDetailedMap_, 
    apvErrorDetailedMap_, 
    apvAddressErrorDetailedMap_, 
    unlockedDetailedMap_, 
    outOfSyncDetailedMap_;


  HistogramConfig fedErrorsVsId_;

  //has individual FED histogram been booked? (index is FedId)
  std::vector<bool> histosBooked_, 
    debugHistosBooked_;

  HistogramConfig tkMapConfig_;
  TkHistoMap *tkmapFED_;

  HistogramConfig lumiErrorFraction_;


};//class



#endif //DQM_SiStripMonitorHardware_FEDHistograms_HH
