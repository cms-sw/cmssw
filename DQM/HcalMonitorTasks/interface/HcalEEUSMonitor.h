#ifndef GUARD_DQM_HCALMONITORTASKS_HCALEEUSMONITOR_H
#define GUARD_DQM_HCALMONITORTASKS_HCALEEUSMONITOR_H

#define  NUMSPIGS 15
#define  NUMFEDS  32
#define  NUMCHANS 24


#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
// The following are needed for using pedestals in fC:
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"

// Raw data stuff
#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

// Use for stringstream
#include <iostream>
#include <iomanip>
#include <cmath>

/** \class HcalEEUSMonitor
  *
  * $Date: 2009/07/06 10:51:54 $
  * $Revision: 1.3 $
  * \author J. Temple - Univ. of Maryland
  */

class HcalEEUSMonitor:  public HcalBaseMonitor {
 public:
  HcalEEUSMonitor();
  ~HcalEEUSMonitor();

  void unpack(const FEDRawData& raw, const HcalElectronicsMap& emap);
  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void reset();
  void clearME();

  // processEvent routine -- specifies what inputs are looked at each event
  void processEvent(const FEDRawDataCollection& rawraw,
		    const HcalUnpackerReport& report,
		    const HcalElectronicsMap& emap
		    //const ZDCRecHitCollection& zdcHits
		    );
  // Check Raw Data each event
  void processEvent_RawData(const FEDRawDataCollection& rawraw,
			    const HcalUnpackerReport& report,
			    const HcalElectronicsMap& emap);

 private:
  
  int ievt_;
  MonitorElement* meEVT_;

  std::vector <int> fedUnpackList_;
  int firstFED_;


  //Jason's MEs

  MonitorElement* meEECorrel_;
  MonitorElement* meEEPerSpigot_;
  MonitorElement* meEEThisEvent_;

  //Jason's Variables

  bool EEthisEvent [NUMSPIGS * NUMFEDS];  //Bookkeeping: which spigots EE
  int numEEthisEvent;
  //Francesco's MEs

  MonitorElement* meNormFractSpigs_US0_EE0_;
  MonitorElement* meEEFractSpigs_US0_EE1_;
  MonitorElement* meUSFractSpigs_US1_EE0_;
  MonitorElement* meUSFractSpigs_US1_EE1_;

  MonitorElement* meRawDataLength2_US0_EE0_;
  MonitorElement* meRawDataLength2_US0_EE1_;
  MonitorElement* meRawDataLength2_US1_EE0_;
  MonitorElement* meRawDataLength2_US1_EE1_;

  //---------
  //Francesco's Variables
  uint64_t UScount[NUMFEDS][NUMSPIGS];
  uint64_t US0EE0count[NUMFEDS][NUMSPIGS];
  uint64_t US0EE1count[NUMFEDS][NUMSPIGS];
  uint64_t US1EE0count[NUMFEDS][NUMSPIGS];
  uint64_t US1EE1count[NUMFEDS][NUMSPIGS];

  //Ted's MEs
  //Ted's Variables

  //Jared's MEs

  MonitorElement* meNumberEETriggered_;//[NUMFEDS];
  MonitorElement* meNumberNETriggered_;//[NUMFEDS];
  MonitorElement* meNumberTriggered_;//[NUMFEDS];

  //Jared's Variables
  uint32_t consecutiveEETriggers[NUMFEDS][NUMSPIGS];
  uint32_t consecutiveNETriggers[NUMFEDS][NUMSPIGS];
  uint32_t consecutiveTriggers[NUMFEDS][NUMSPIGS];
  int dccOrN;
  int prevOrN;
  int prevWasEE[NUMFEDS][NUMSPIGS];
}; // class HcalEEUSMonitor

#endif  
