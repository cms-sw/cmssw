#ifndef DQM_HCALMONITORTASKS_HCALDATAFORMATMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDATAFORMATMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

/** \class Hcaldataformatmonitor
 *
 * $Date: 2007/04/20 15:11:10 $
 * $Revision: 1.12 $
 * \author W. Fisher - FNAL
 */
class HcalDataFormatMonitor: public HcalBaseMonitor {
 public:
  HcalDataFormatMonitor();
  ~HcalDataFormatMonitor();
  
  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const FEDRawDataCollection& rawraw, const
		    HcalUnpackerReport& report, const HcalElectronicsMap& emap);
  void unpack(const FEDRawData& raw, const HcalElectronicsMap& emap);
  void clearME();
  
 private: // Data accessors
   vector<int> fedUnpackList_;
   int firstFED_;
   int ievt_;
   int lastEvtN_;
   
 private:  //Monitoring elements
   
   MonitorElement* meEVT_;
   
   //MEs for hcalunpacker report info
   MonitorElement* meSpigotFormatErrors_;
   MonitorElement* meBadQualityDigis_;
   MonitorElement* meUnmappedDigis_;
   MonitorElement* meUnmappedTPDigis_;
   MonitorElement* meFEDerrorMap_;
   //Check that evt numbers are synchronized across all HTRs
   MonitorElement* meEvtNumberSynch_;
   
   // The following MEs map specific conditons from the HTR/DCC headers as specified in
   //   http://cmsdoc.cern.ch/cms/HCAL/document/CountingHouse/HTR/design/Rev4MainFPGA.pdf
   struct{
     MonitorElement* DCC_ErrWd;  //16 (12??) bit HTR error word, Ext. Header 3
     MonitorElement* SpigotMap;  //Map of HTR errors into Spigot/DCCID space
     MonitorElement* CrateMap;   //Map of HTR errors into Crate/Slot space
     MonitorElement* ExtHeader5; //16 bits from Ext. Header 5
     MonitorElement* ExtHeader7; //16 bits from Ext. Header 7     
     //Map of into HTR error bits into Spigot/DCCID space
     //  listed in increasing order from LSB of Byte 0
     MonitorElement* OWMap;      //Overflow warnings
     MonitorElement* BZMap;      //Internal buffer busy
     MonitorElement* EEMap;      //Empty event
     MonitorElement* RLMap;      //Reject L1A
     MonitorElement* LEMap;      //Latency errors 
     MonitorElement* LWMap;      //Latency warnings 
     MonitorElement* ODMap;      //Optical Data errors 
     MonitorElement* CKMap;      //Clock errors 
     MonitorElement* BEMap;      //Bunch error
   } hbHists,heHists,hfHists,hoHists;
   
};

#endif
