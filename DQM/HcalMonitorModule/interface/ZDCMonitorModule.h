#ifndef ZDCMonitorModule_GUARD_H
#define ZDCMonitorModule_GUARD_H

/*
 * \file ZDCMonitorModule.h
 *

 * $Date: 2010/03/25 20:00:36 $
 * $Revision: 1.2 $
 * \author 
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

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "FWCore/Utilities/interface/CPUTimer.h"

class MonitorElement;
class DQMStore;
class  HcalZDCMonitor;

#include <iostream>
#include <fstream>

class ZDCMonitorModule : public edm::EDAnalyzer{

public:
  
  // Constructor
  ZDCMonitorModule(const edm::ParameterSet& ps);

  // Destructor
  ~ZDCMonitorModule();
  
 protected:
  
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  // BeginJob
  void beginJob();
  
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
/*
  // Check ZDC has FED data
  void CheckZDCStatus	(const FEDRawDataCollection& rawraw,
                      	const HcalUnpackerReport& report,
                      	const HcalElectronicsMap& emap,
                      	const ZDCDigiCollection& zdcdigi
                      	);
*/
 private:
  std::vector<int> fedss;
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

  // Reset histograms every N events

  /// The name of the monitoring process which derives from this
  /// class, used to standardize filename and file structure
  std::string monitorName_;

  /// Verbosity switch used for debugging or informational output
  int debug_;  // make debug an int in order to allow different levels of messaging

  // control whether or not to display time used by each module
  bool showTiming_; 
  edm::CPUTimer cpu_timer; // 

  // counters and flags
  int nevt_;

  struct{
    timeval startTV,updateTV;
    double elapsedTime; 
    double vetoTime; 
    double updateTime;
  } psTime_;    

  ///Connection to the DQM backend
  DQMStore* dbe_;  
  
  // environment variables
  int irun_,ievent_,itime_;
  unsigned int ilumisec;
  bool Online_;
  std::string rootFolder_;

  int ievt_;
  int ievt_rawdata_;
  int ievt_digi_;
  int ievt_rechit_;
  int ievt_pre_; // copy of counter used for prescale purposes
  bool fedsListed_;
  
  edm::InputTag inputLabelDigi_;
  edm::InputTag inputLabelRecHitZDC_;

  edm::InputTag FEDRawDataCollection_; // not yet in use, but we still store the tag name

  MonitorElement* meIEVTALL_;
  MonitorElement* meIEVTRAW_;
  MonitorElement* meIEVTDIGI_;
  MonitorElement* meIEVTRECHIT_;

  MonitorElement* meFEDS_;
  MonitorElement* meStatus_;
  MonitorElement* meTrigger_;
  MonitorElement* meLatency_;
  MonitorElement* meQuality_;
  
  HcalZDCMonitor*         zdcMon_;

  ////---- decide whether the ZDC status should be checked
  bool checkZDC_;

  edm::ESHandle<HcalDbService> conditions_;
  const HcalElectronicsMap*    readoutMap_;

  std::ofstream m_logFile;

  // Determine whether the ZDC in the run (using FED info)
  int ZDCpresent_;
  MonitorElement* meZDC_;

  // myquality_ will store status values for each det ID I find
  bool dump2database_;
  //std::map<HcalDetId, unsigned int> myquality_;
  //HcalChannelQuality* chanquality_;
};

#endif
