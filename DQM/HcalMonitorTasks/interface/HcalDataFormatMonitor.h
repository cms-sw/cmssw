#ifndef DQM_HCALMONITORTASKS_HCALDATAFORMATMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDATAFORMATMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

/** \class Hcaldataformatmonitor
  *  
  * $Date: 2007/03/21 23:01:29 $
  * $Revision: 1.10 $
  * \author W. Fisher - FNAL
  */
class HcalDataFormatMonitor: public HcalBaseMonitor {
public:
  HcalDataFormatMonitor(); 
  ~HcalDataFormatMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const FEDRawDataCollection& rawraw, const HcalUnpackerReport& report, const HcalElectronicsMap& emap);
  void unpack(const FEDRawData& raw, const HcalElectronicsMap& emap);
  void clearME();

private: /// Data accessors
  vector<int> fedUnpackList_;
  int firstFED_;
  int ievt_;

private:  ///Monitoring elements

  MonitorElement* meEVT_;

  MonitorElement* meSpigotFormatErrors_;
  MonitorElement* meBadQualityDigis_;
  MonitorElement* meUnmappedDigis_;
  MonitorElement* meUnmappedTPDigis_;
  MonitorElement* meFEDerrorMap_;
  
  struct{
    MonitorElement* ERR_MAP;
    MonitorElement* DCC_ERRWD;
    MonitorElement* FiberMap;
    MonitorElement* SpigotMap;
  } hbHists, hfHists,hoHists;

  

};

#endif
