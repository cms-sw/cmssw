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

  ~CMHistograms();
  
  //initialise histograms
  void initialise(const edm::ParameterSet& iConfig,
		  std::ostringstream* pDebugStream
		  );

  void fillHistograms(std::vector<CMvalues> aVec, float aTime, unsigned int aFedId);


   //book the top level histograms
  void bookTopLevelHistograms(DQMStore* dqm);

  //book individual FED histograms or book all FED level histograms at once
  void bookFEDHistograms(unsigned int fedId);
  //void bookFEDHistograms(unsigned int fedId, unsigned int aCategory);
  void bookChannelsHistograms(unsigned int fedId);

  void bookAllFEDHistograms();

  std::string tkHistoMapName(unsigned int aIndex=0);

  TkHistoMap * tkHistoMapPointer(unsigned int aIndex=0);

protected:
  
private:

  //has individual FED histogram been booked? (index is FedId)
  std::vector<bool> histosBooked_;

  bool doFed_[500];

  std::string tkMapConfigName_;
  TkHistoMap *tkmapCM_[4];

  MonitorElement *medianAPV1vsAPV0_;
  MonitorElement *medianAPV0minusAPV1_;

  MonitorElement *meanCMPerFedvsFedId_;
  MonitorElement *meanCMPerFedvsTime_;

  //CM-previous value for all APVs
  MonitorElement *variationsPerFedvsFedId_;
  MonitorElement *variationsPerFedvsTime_;

  std::map<unsigned int,MonitorElement*> medianAPV1vsAPV0perFED_;
  std::map<unsigned int,MonitorElement*> medianAPV0minusAPV1perFED_;
  std::map<unsigned int,std::vector<MonitorElement*> > medianperChannel_;
  std::map<unsigned int,std::vector<MonitorElement*> > medianAPV0minusAPV1perChannel_;

};//class



#endif //DQM_SiStripMonitorHardware_CMHistograms_HH
