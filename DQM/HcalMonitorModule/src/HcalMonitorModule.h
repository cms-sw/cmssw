#ifndef HcalMonitorModule_H
#define HcalMonitorModule_H

/*
 * \file HcalMonitorModule.h
 *
 * $Date: 2007/11/21 20:45:14 $
 * $Revision: 1.22 $
 * \author W. Fisher
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
//#include "DQMServices/Components/interface/DQMAnalyzer.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/Provenance/interface/EventID.h"  
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DQM/HcalMonitorModule/interface/HcalMonitorSelector.h"
#include "DQM/HcalMonitorTasks/interface/HcalDigiMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalDataFormatMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalRecHitMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalPedestalMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalLEDMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalMTCCMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalHotCellMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalDeadCellMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalTrigPrimMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalTemplateAnalysis.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/time.h>

using namespace std;
using namespace edm;

class HcalMonitorModule : public EDAnalyzer{

public:
  
  // Constructor
  HcalMonitorModule(const edm::ParameterSet& ps);

  // Destructor
  ~HcalMonitorModule();
  
 protected:
  
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  // BeginJob
  void beginJob(const edm::EventSetup& c);
  
  // BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& c);

  // Begin LumiBlock
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& c) ;

  // End LumiBlock
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  // EndJob
  void endJob(void);
  
  // EndRun
  void endRun(const edm::Run& run, const edm::EventSetup& c);

  // Reset
  void reset(void);

  /// Boolean prescale test for this event
  bool prescale();

 private:
  /********************************************************/
  //  The following member variables can be specified in  //
  //  the configuration input file for the process.       //
  /********************************************************/

  /// Prescale variables for restricting the frequency of analyzer
  /// behavior.  The base class does not implement prescales.
  /// Set to -1 to be ignored.
  int prescaleEvt_;    ///units of events
  int prescaleLS_;     ///units of lumi sections
  int prescaleTime_;   ///units of minutes
  int prescaleUpdate_; ///units of "updates", TBD

  /// The name of the monitoring process which derives from this
  /// class, used to standardize filename and file structure
  std::string monitorName_;

  /// Verbosity switch used for debugging or informational output
  bool debug_;

  /// counters and flags
  int nevt_;
  int nlumisecs_;
  bool saved_;

  struct{
    timeval startTV,updateTV;
    double elapsedTime; 
    double vetoTime; 
    double updateTime;
  } psTime_;    

  ///Connection to the DQM backend
  DaqMonitorBEInterface* dbe_;  
  
  // environment variables
  int irun_,ilumisec_,ievent_,itime_;
  bool actonLS_ ;
  std::string rootFolder_;

  int ievt_;
  bool fedsListed_;
  
  edm::InputTag inputLabelGT_;
  edm::InputTag inputLabelDigi_;
  edm::InputTag inputLabelRecHitHBHE_;
  edm::InputTag inputLabelRecHitHF_;
  edm::InputTag inputLabelRecHitHO_;

  MonitorElement* meFEDS_;
  MonitorElement* meStatus_;
  MonitorElement* meRunType_;
  MonitorElement* meEvtMask_;
  MonitorElement* meTrigger_;
  MonitorElement* meLatency_;
  MonitorElement* meQuality_;
  

  HcalMonitorSelector*    evtSel_;
  HcalDigiMonitor*        digiMon_;
  HcalDataFormatMonitor*  dfMon_;
  HcalRecHitMonitor*      rhMon_;
  HcalPedestalMonitor*    pedMon_;
  HcalLEDMonitor*         ledMon_;
  HcalMTCCMonitor*        mtccMon_;
  HcalHotCellMonitor*     hotMon_;
  HcalDeadCellMonitor*    deadMon_;
  HcalTrigPrimMonitor*    tpMon_;
  HcalTemplateAnalysis*   tempAnalysis_;
  
  edm::ESHandle<HcalDbService> conditions_;
  const HcalElectronicsMap*    readoutMap_;

};

#endif
