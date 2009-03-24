#ifndef HcalMonitorModule_H
#define HcalMonitorModule_H

/*
 * \file HcalMonitorModule.h
 *
 * $Date: 2009/02/11 18:36:29 $
 * $Revision: 1.41 $
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


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DQMServices/Components/interface/DQMAnalyzer.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Utilities/interface/CPUTimer.h"

#include "DataFormats/Provenance/interface/EventID.h"  
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DQM/HcalMonitorModule/interface/HcalMonitorSelector.h"
#include "DQM/HcalMonitorTasks/interface/HcalDigiMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalDataFormatMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalDataIntegrityTask.h"
#include "DQM/HcalMonitorTasks/interface/HcalRecHitMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalBeamMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalExpertMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalPedestalMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalLEDMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalLaserMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalMTCCMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalHotCellMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalDeadCellMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalCaloTowerMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalTrigPrimMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalTemplateAnalysis.h"
#include "DQM/HcalMonitorTasks/interface/HcalEEUSMonitor.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"

// Use to hold/get channel status
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/time.h>

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

  // Check which subdetectors have FED data
  void CheckSubdetectorStatus(const FEDRawDataCollection& rawraw, 
			      const HcalUnpackerReport& report, 
			      const HcalElectronicsMap& emap,
			      const HBHEDigiCollection& hbhedigi,
			      const HODigiCollection& hodigi,
			      const HFDigiCollection& hfdigi
			      //const ZDCDigiCollection& zdcdigi,
			      );
    
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
  int nlumisecs_;
  bool saved_;

  struct{
    timeval startTV,updateTV;
    double elapsedTime; 
    double vetoTime; 
    double updateTime;
  } psTime_;    

  ///Connection to the DQM backend
  DQMStore* dbe_;  
  
  // environment variables
  int irun_,ilumisec_,ievent_,itime_;
  bool actonLS_ ;
  std::string rootFolder_;

  int ievt_;
  int ievt_pre_; // copy of counter used for prescale purposes
  bool fedsListed_;
  
  edm::InputTag inputLabelGT_;
  edm::InputTag inputLabelDigi_;
  edm::InputTag inputLabelRecHitHBHE_;
  edm::InputTag inputLabelRecHitHF_;
  edm::InputTag inputLabelRecHitHO_;
  edm::InputTag inputLabelRecHitZDC_;

  edm::InputTag inputLabelCaloTower_;
  edm::InputTag inputLabelLaser_;
  edm::InputTag FEDRawDataCollection_;

  // Maps of readout hardware unit to calorimeter channel
  std::map<uint32_t, std::vector<HcalDetId> > DCCtoCell;
  std::map<uint32_t, std::vector<HcalDetId> > ::iterator thisDCC;
  std::map<pair <int,int> , std::vector<HcalDetId> > HTRtoCell;
  std::map<pair <int,int> , std::vector<HcalDetId> > ::iterator thisHTR;

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
  HcalDataIntegrityTask*  diTask_;
  HcalRecHitMonitor*      rhMon_;
  HcalBeamMonitor*        beamMon_;
  HcalExpertMonitor*      expertMon_;
  HcalPedestalMonitor*    pedMon_;
  HcalLEDMonitor*         ledMon_;
  HcalLaserMonitor*       laserMon_;
  HcalMTCCMonitor*        mtccMon_;
  HcalHotCellMonitor*     hotMon_;
  HcalDeadCellMonitor*    deadMon_;
  HcalCaloTowerMonitor*   ctMon_;
  HcalTrigPrimMonitor*    tpMon_;
  HcalTemplateAnalysis*   tempAnalysis_;
  HcalEEUSMonitor*        eeusMon_;

  edm::ESHandle<HcalDbService> conditions_;
  const HcalElectronicsMap*    readoutMap_;

  ofstream m_logFile;

  // Running on the Orbit Gap Calibration events?
  bool AnalyzeOrbGapCT_;

  // Decide whether individual subdetectors should be checked
  bool checkHB_;
  bool checkHE_;
  bool checkHO_;
  bool checkHF_;
  bool checkZDC_; // not yet implemented 

  // Determine which subdetectors are in the run (using FED info)
  int HBpresent_;
  int HEpresent_;
  int HOpresent_;
  int HFpresent_;
  int ZDCpresent_; // need to implement
  MonitorElement* meHB_;
  MonitorElement* meHE_;
  MonitorElement* meHO_;
  MonitorElement* meHF_;
  MonitorElement* meZDC_;

  // myquality_ will store status values for each det ID I find
  bool dump2database_;
  std::map<HcalDetId, unsigned int> myquality_;
  HcalChannelQuality* chanquality_;
};

#endif
