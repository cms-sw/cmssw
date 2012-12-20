#ifndef CastorMonitorModule_H
#define CastorMonitorModule_H

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

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimG4CMS/Calo/interface/CaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "FWCore/Utilities/interface/CPUTimer.h"
#include "DataFormats/Provenance/interface/EventID.h"  

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"


#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorCluster.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "DataFormats/JetReco/interface/CastorJetID.h"
#include "RecoJets/JetProducers/interface/CastorJetIDHelper.h"
#include "RecoJets/JetProducers/plugins/CastorJetIDProducer.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h" //-- no CastorUnpackerReport at the moment !
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h" //-- HcalCastorDetId

#include "DQM/CastorMonitor/interface/CastorMonitorSelector.h"
#include "DQM/CastorMonitor/interface/CastorDigiMonitor.h"
#include "DQM/CastorMonitor/interface/CastorRecHitMonitor.h"
#include "DQM/CastorMonitor/interface/CastorChannelQualityMonitor.h"
#include "DQM/CastorMonitor/interface/CastorLEDMonitor.h"
#include "DQM/CastorMonitor/interface/CastorPSMonitor.h"
#include "DQM/CastorMonitor/interface/CastorEventDisplay.h"
#include "DQM/CastorMonitor/interface/CastorHIMonitor.h"
#include "DQM/CastorMonitor/interface/CastorDataIntegrityMonitor.h"
#include "DQM/CastorMonitor/interface/CastorTowerJetMonitor.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h" //-- use to get/dump Calib to DB 
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h" //-- use to get/hold channel status
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"


//// #include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h" //-- 

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/time.h>




class CastorMonitorModule : public edm::EDAnalyzer{

public:
  
  ////---- constructor
  CastorMonitorModule(const edm::ParameterSet& ps);

  ////---- destructor
  ~CastorMonitorModule();
  
 protected:
  
  ////---- analyze
  void analyze(const edm::Event& iEvent, const edm::EventSetup& eventSetup);
  
  ////---- beginJob
  void beginJob();
  
  ////---- beginRun
  void beginRun(const edm::Run& iRun, const edm::EventSetup& eventSetup);

  ////---- begin LumiBlock
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& eventSetup) ;

  ////---- end LumiBlock
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& eventSetup);

  ////---- endJob
  void endJob(void);
  
  ////---- endRun
  void endRun(const edm::Run& run, const edm::EventSetup& eventSetup);

  ////---- reset
  void reset(void);

  ////---- boolean prescale test for event
  bool prescale();

  ////---- check whether Castor has FED data
  void CheckCastorStatus     (const FEDRawDataCollection& rawraw, 
			      const HcalUnpackerReport& report, 
			      const CastorElectronicsMap& emap,
			      const CastorDigiCollection& castordigi
			      );
    
 private:
 
  ////----
  ////---- steerable variables that can be specified in the configuration 
  ////---- input file for the process.       
  ////----
  ////---- prescale variables for restricting the frequency of analyzer
  ////---- behavior.  The base class does not implement prescales.
  ////---- set to -1 to be ignored.
  int prescaleEvt_;    //-- units of events
  int prescaleLS_;     //-- units of lumi sections
  int prescaleTime_;   //-- units of minutes
  int prescaleUpdate_; //-- units of "updates", TBD

  ////---- name of the monitoring process which derives from this
  ////---- class, used to standardize filename and file structure
  std::string monitorName_;

  ////---- verbosity switch used for debugging or informational output
  int fVerbosity;  

  ////---- counters and flags
  int nevt_;
  int nlumisecs_;
  bool saved_;

  ////---- castor products among the event data   
  bool rawOK_    ;
  bool reportOK_ ;
  bool digiOK_   ;
  bool rechitOK_ ;
  bool towerOK_  ;
  bool jetOK_    ;
  bool jetIdOK_  ;
  
  int nRaw;
  int nDigi;
  int nRechit;
  int nTower;
  int nJet;
  int nJetId;
   
  
  ////---- control whether or not to display time used by each module
  bool showTiming_; 
  edm::CPUTimer cpu_timer; 

  ////---- psTime
  struct{
    timeval startTV,updateTV;
    double elapsedTime; 
    double vetoTime; 
    double updateTime;
  } psTime_;    


  ////---- define the DQMStore 
  DQMStore* dbe_;  
  
  ////---- define environment variables
  int irun_,ilumisec_,ievent_,itime_,ibunch_;
  bool actonLS_ ;
  std::string rootFolder_;

  int ievt_;
  int ievt_pre_; //-- copy of counter used for prescale purposes
  bool fedsListed_;
  
  //edm::InputTag inputLabelGT_;
  edm::InputTag inputLabelRaw_;
  edm::InputTag inputLabelReport_;
  edm::InputTag inputLabelDigi_;
  edm::InputTag inputLabelRecHitCASTOR_;
  edm::InputTag inputLabelTowerCASTOR_;  
  edm::InputTag inputLabelBasicJetCASTOR_;  
  edm::InputTag inputLabelJetIdCASTOR_ ;
  edm::InputTag inputLabelCastorTowers_    ; 
  edm::InputTag inputLabelCastorBasicJets_ ; 
  edm::InputTag inputLabelCastorJetIDs_ ; 



  ////---- define  CastorTowerCollection
  // typedef std::vector<reco::CastorTower> CastorTowerCollection;

  //edm::InputTag inputLabelCaloTower_;
  //edm::InputTag inputLabelLaser_;

  ////---- Maps of readout hardware unit to calorimeter channel
  std::map<uint32_t, std::vector<HcalCastorDetId> > DCCtoCell;
  std::map<uint32_t, std::vector<HcalCastorDetId> > ::iterator thisDCC;
  std::map<std::pair <int,int> , std::vector<HcalCastorDetId> > HTRtoCell;
  std::map<std::pair <int,int> , std::vector<HcalCastorDetId> > ::iterator thisHTR;

  ////---- define ME used to display the DQM Job Status
  MonitorElement* meFEDS_;
  MonitorElement* meStatus_;
  MonitorElement* meRunType_;
  MonitorElement* meEvtMask_;
  MonitorElement* meTrigger_;
  MonitorElement* meLatency_;
  MonitorElement* meQuality_;
  MonitorElement* CastorEventProduct;
  //MonitorElement* CastorMonitorStatus;
  

  ////---- define monitors
  CastorMonitorSelector*    evtSel_;
  CastorRecHitMonitor*      RecHitMon_;
  CastorChannelQualityMonitor*  CQMon_;
  CastorDigiMonitor*        DigiMon_;
  CastorLEDMonitor*         LedMon_;
  CastorPSMonitor*          PSMon_;
  CastorEventDisplay*       EDMon_;
  CastorHIMonitor*          HIMon_;
  CastorDataIntegrityMonitor* DataIntMon_;
  CastorTowerJetMonitor*     TowerJetMon_;

  MonitorElement* meEVT_;

  edm::ESHandle<CastorDbService> conditions_;
  const CastorElectronicsMap*     CastorReadoutMap_;

  ////---- pedestal parameters from CastorPedestalsRcd, initialized in beginRun
  edm::ESHandle<CastorPedestals> dbPedestals;

  // pedestal width averaged over capIDs, calculated in beginRun
  // aware of the difference between index[0..15][0..13] 
  // and sector/module numeration[1..16][1..14]
  float        fPedestalNSigmaAverage[14][16];

  std::vector<HcalGenericDetId> listEMap; //electronics Emap


  ofstream m_logFile;

  ////---- decide whether the Castor status should be checked
  bool checkCASTOR_;

  ////----- define this ME to check whether the Castor is present 
  ////----- in the run (using FED info)  
  ////----- 1 is present , 0 - no Digis , -1 no within FED 
  MonitorElement* meCASTOR_;
   
  ////---- to remove the EventDisplay Monitor in the Offline case
  bool EDMonOn_;

  /////---- myquality_ will store status values for each det ID I find
  bool dump2database_;
  std::map<HcalCastorDetId, unsigned int> myquality_;
  CastorChannelQuality* chanquality_;
};

#endif
