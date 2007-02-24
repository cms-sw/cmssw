#ifndef DQM_HCALMONITORTASKS_HCALDATAFORMATMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDATAFORMATMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDataFrameFilter.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "TH1F.h"
#include "DQMServices/CoreROOT/interface/MonitorElementRootT.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

/** \class Hcaldataformatmonitor
  *  
  * $Date: 2006/12/12 19:10:27 $
  * $Revision: 1.8 $
  * \author W. Fisher - FNAL
  */
class HcalDataFormatMonitor: public HcalBaseMonitor {
public:
  HcalDataFormatMonitor(); 
  ~HcalDataFormatMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const FEDRawDataCollection& rawraw, const HcalElectronicsMap& emap);
  void unpack(const FEDRawData& raw, const HcalElectronicsMap& emap);
  void clearME();

private: /// Data accessors
  vector<int> fedUnpackList_;
  int firstFED_;
  int ievt_;

private:  ///Monitoring elements

  MonitorElement* meEVT_;

  struct{
    MonitorElement* ERR_MAP;
    MonitorElement* DCC_ERRWD;
    MonitorElement* FiberMap;
    MonitorElement* SpigotMap;
    //    HcalElectronicsId eid(fc,f,spigot,dccid);
    //    eid.setHTR(htr.readoutVMECrateId(),htr.htrSlot(),htr.htrTopBottom());
  } hbHists, hfHists,hoHists;

};

#endif
