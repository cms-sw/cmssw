#ifndef DQM_HCALMONITORTASKS_HCALDATAFORMATMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDATAFORMATMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

/** \class Hcaldataformatmonitor
 *
 * $Date: 2007/06/09 21:12:35 $
 * $Revision: 1.13 $
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
   vector<int> dccCrate_;
   vector<HcalSubdetector> dccSubdet_;
   int firstFED_;
   int ievt_;
   int lastEvtN_;
   //   int dccnum_;
   //int cratenum_;
   
   bool dumpBCN_;
   int bcnCrate_;
     
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
   MonitorElement* meErrWdCrate_;  //HTR error bits by crate

   MonitorElement* meCrate0HTRErr_;   //Map of HTR errors into Crate 0
   MonitorElement* meCrate1HTRErr_;   //Map of HTR errors into Crate 1
   MonitorElement* meCrate2HTRErr_;   //Map of HTR errors into Crate 2
   MonitorElement* meCrate3HTRErr_;   //Map of HTR errors into Crate 3
   MonitorElement* meCrate4HTRErr_;   //Map of HTR errors into Crate 4
   MonitorElement* meCrate5HTRErr_;   //Map of HTR errors into Crate 5
   MonitorElement* meCrate6HTRErr_;   //Map of HTR errors into Crate 6
   MonitorElement* meCrate7HTRErr_;   //Map of HTR errors into Crate 7
   MonitorElement* meCrate8HTRErr_;   //Map of HTR errors into Crate 8
   MonitorElement* meCrate9HTRErr_;   //Map of HTR errors into Crate 9
   MonitorElement* meCrate10HTRErr_;   //Map of HTR errors into Crate 10
   MonitorElement* meCrate11HTRErr_;   //Map of HTR errors into Crate 11
   MonitorElement* meCrate12HTRErr_;   //Map of HTR errors into Crate 12
   MonitorElement* meCrate13HTRErr_;   //Map of HTR errors into Crate 13
   MonitorElement* meCrate14HTRErr_;   //Map of HTR errors into Crate 14
   MonitorElement* meCrate15HTRErr_;   //Map of HTR errors into Crate 15
   MonitorElement* meCrate16HTRErr_;   //Map of HTR errors into Crate 16
   MonitorElement* meCrate17HTRErr_;   //Map of HTR errors into Crate 17

   MonitorElement* meCrate0HTRBCN_;   //Map of HTR BCN into Crate 0
   MonitorElement* meCrate1HTRBCN_;   //Map of HTR BCN into Crate 1
   MonitorElement* meCrate2HTRBCN_;   //Map of HTR BCN into Crate 2
   MonitorElement* meCrate3HTRBCN_;   //Map of HTR BCN into Crate 3
   MonitorElement* meCrate4HTRBCN_;   //Map of HTR BCN into Crate 4
   MonitorElement* meCrate5HTRBCN_;   //Map of HTR BCN into Crate 5
   MonitorElement* meCrate6HTRBCN_;   //Map of HTR BCN into Crate 6
   MonitorElement* meCrate7HTRBCN_;   //Map of HTR BCN into Crate 7
   MonitorElement* meCrate8HTRBCN_;   //Map of HTR BCN into Crate 8
   MonitorElement* meCrate9HTRBCN_;   //Map of HTR BCN into Crate 9
   MonitorElement* meCrate10HTRBCN_;   //Map of HTR BCN into Crate 10
   MonitorElement* meCrate11HTRBCN_;   //Map of HTR BCN into Crate 11
   MonitorElement* meCrate12HTRBCN_;   //Map of HTR BCN into Crate 12
   MonitorElement* meCrate13HTRBCN_;   //Map of HTR BCN into Crate 13
   MonitorElement* meCrate14HTRBCN_;   //Map of HTR BCN into Crate 14
   MonitorElement* meCrate15HTRBCN_;   //Map of HTR BCN into Crate 15
   MonitorElement* meCrate16HTRBCN_;   //Map of HTR BCN into Crate 16
   MonitorElement* meCrate17HTRBCN_;   //Map of HTR BCN into Crate 17

   // The following MEs map specific conditons from the HTR/DCC headers as specified in
   //   http://cmsdoc.cern.ch/cms/HCAL/document/CountingHouse/HTR/design/Rev4MainFPGA.pdf

   struct{
     MonitorElement* DCC_ErrWd;  //16 (12??) bit HTR error word, Ext. Header 3
     MonitorElement* SpigotMap;  //Map of HTR errors into Spigot/DCCID space
     MonitorElement* CrateMap;   //Map of HTR errors into Crate/Slot space
     MonitorElement* ExtHeader5; //16 bits from Ext. Header 5
     MonitorElement* ExtHeader7; //16 bits from Ext. Header 7

     /*
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
     */
   } hbheHists,hfHists,hoHists;
   
};

#endif
