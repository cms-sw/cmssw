// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      CMHistograms
// 
/**\class CMHistograms DQM/SiStripMonitorHardware/interface/CMHistograms.hh

 Description: DQM source application to produce data integrety histograms for SiStrip data
*/
//
// Original Author:  Nicholas Cripps in plugin file
//         Created:  2008/09/16
// Modified by    :  Anne-Marie Magnan, code copied from plugin to this class
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
    bool IsShot;
    uint16_t Length;
    std::pair<uint16_t,uint16_t> Medians;
    std::pair<float,float> ShotMedians;
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
  void bookFEDHistograms(unsigned int fedId, unsigned int aCategory);

  void bookAllFEDHistograms();

  std::string tkHistoMapName(unsigned int aIndex=0);

  TkHistoMap * tkHistoMapPointer(unsigned int aIndex=0);

protected:
  
private:

  //has individual FED histogram been booked? (index is FedId)
  std::vector<bool> histosBooked_;

  std::string tkMapConfigName_;
  TkHistoMap *tkmapCM_[4];

  MonitorElement *medianAPV0_[5];
  MonitorElement *medianAPV1_[5];
  MonitorElement *medianAPV0vsTime_[5];
  MonitorElement *medianAPV1vsTime_[5];

  MonitorElement *shotMedianAPV0_;
  MonitorElement *shotMedianAPV1_;
  MonitorElement *medianAPV1vsAPV0_[5];
  MonitorElement *medianAPV1minusAPV0_[5];
  MonitorElement *medianAPV1minusAPV0vsTime_[5];
  MonitorElement *medianAPV1minusAPV0minusShotMedianAPV1_;
  MonitorElement *medianAPV0minusAPV1minusShotMedianAPV1_;

  std::map<unsigned int,MonitorElement*> medianAPV1vsAPV0perFED_[5];
  std::map<unsigned int,MonitorElement*> medianAPV1minusAPV0perFED_[5];
  //std::map<unsigned int,MonitorElement*> medianAPV1minusAPV0vsTimeperFED_[5];
  //std::map<unsigned int,MonitorElement*> medianAPV0vsTimeperFED_[5];
  //std::map<unsigned int,MonitorElement*> medianAPV1vsTimeperFED_[5];

  std::string categories_[5]; 

};//class



#endif //DQM_SiStripMonitorHardware_CMHistograms_HH
