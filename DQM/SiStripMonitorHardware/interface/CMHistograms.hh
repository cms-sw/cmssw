// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      CMHistograms
// 
/**\class CMHistograms DQM/SiStripMonitorHardware/interface/CMHistograms.hh

 Description: DQM source application to produce CM monitoring histograms for SiStrip data
*/
//
// Original Author:  Anne-Marie Magnan
//         Created:  2009/07/22
//

#ifndef DQM_SiStripMonitorHardware_CMHistograms_HH
#define DQM_SiStripMonitorHardware_CMHistograms_HH

#include <sstream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripMonitorHardware/interface/HistogramBase.hh"

class CMHistograms: public HistogramBase {

public:

  struct CMvalues {
    unsigned int ChannelID;
    //bool IsShot;
    //uint16_t Length;
    std::pair<uint16_t,uint16_t> Medians;
    //std::pair<float,float> ShotMedians;
    std::pair<uint16_t,uint16_t> PreviousMedians;
  };

  CMHistograms();

  ~CMHistograms() override;
  
  //initialise histograms
  void initialise(const edm::ParameterSet& iConfig,
		  std::ostringstream* pDebugStream
		  ) override;

  void fillHistograms(const std::vector<CMvalues>& aVec, float aTime, unsigned int aFedId);


   //book the top level histograms
  void bookTopLevelHistograms(DQMStore::IBooker &);

  //book individual FED histograms or book all FED level histograms at once
  void bookFEDHistograms(DQMStore::IBooker & , unsigned int fedId);
  //void bookFEDHistograms(unsigned int fedId, unsigned int aCategory);
  void bookChannelsHistograms(DQMStore::IBooker & , unsigned int fedId);

  void bookAllFEDHistograms(DQMStore::IBooker &);

  bool tkHistoMapEnabled(unsigned int aIndex=0) override;

  TkHistoMap * tkHistoMapPointer(unsigned int aIndex=0) override;

protected:
  
private:

  //has individual FED histogram been booked? (index is FedId)
  std::vector<bool> histosBooked_;

  bool doFed_[500];

  HistogramConfig tkMapConfig_;
  TkHistoMap *tkmapCM_[4];

  HistogramConfig medianAPV1vsAPV0_;
  HistogramConfig medianAPV0minusAPV1_;

  HistogramConfig meanCMPerFedvsFedId_;
  HistogramConfig meanCMPerFedvsTime_;

  //CM-previous value for all APVs
  HistogramConfig variationsPerFedvsFedId_;
  HistogramConfig variationsPerFedvsTime_;

  HistogramConfig medianAPV1vsAPV0perFED_;
  HistogramConfig medianAPV0minusAPV1perFED_;
  HistogramConfig medianperChannel_;
  HistogramConfig medianAPV0minusAPV1perChannel_;

  std::map<unsigned int,MonitorElement* > medianAPV1vsAPV0perFEDMap_;
  std::map<unsigned int,MonitorElement* > medianAPV0minusAPV1perFEDMap_;
  std::map<unsigned int,std::vector<MonitorElement* > > medianperChannelMap_;
  std::map<unsigned int,std::vector<MonitorElement* > > medianAPV0minusAPV1perChannelMap_;

};//class



#endif //DQM_SiStripMonitorHardware_CMHistograms_HH
