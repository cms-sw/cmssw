#ifndef HcalMonitorModule_GUARD_H
#define HcalMonitorModule_GUARD_H

/*
 * \file HcalMonitorModule.h
 *

 * $Date: 2010/03/25 10:59:12 $
 * $Revision: 1.8 $
 * \author J. Temple
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"

// forward declarations

class DQMStore;
class MonitorElement;
class FEDRawDataCollection;
class HcalElectronicsMap;

class HcalMonitorModule : public edm::EDAnalyzer
{

public:

  // Constructor
  HcalMonitorModule(const edm::ParameterSet& ps);

  // Destructor
  ~HcalMonitorModule();

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

  // cleanup
  void cleanup(void);

  // setup
  void setup(void);
  
  // CheckSubdetectorStatus
  void CheckSubdetectorStatus(const edm::Handle<FEDRawDataCollection>& rawraw,
			      HcalSubdetector subdet,
			      const HcalElectronicsMap& emap);

 private:

  int ievt_;
  int runNumber_;
  int evtNumber_;

  MonitorElement* meStatus_;
  MonitorElement* meRun_;
  MonitorElement* meEvt_;
  MonitorElement* meFEDS_;
  MonitorElement* meCalibType_;
  MonitorElement* meCurrentCalibType_;
  MonitorElement* meHB_;
  MonitorElement* meHE_;
  MonitorElement* meHO_;
  MonitorElement* meHF_;
  MonitorElement* meIevt_;
  MonitorElement* meIevtHist_; 
  MonitorElement* meEvtsVsLS_;
  MonitorElement* meProcessedEndLumi_;
  MonitorElement* meOnline_;

  bool fedsListed_;

  bool Online_;
  bool mergeRuns_;
  bool enableCleanup_;
  int debug_;
  bool init_;
  edm::InputTag FEDRawDataCollection_;
  edm::InputTag inputLabelReport_;
  std::string prefixME_;
  int NLumiBlocks_;

  int HBpresent_, HEpresent_, HOpresent_, HFpresent_;
  DQMStore* dbe_;

  const HcalElectronicsMap*    eMap_;
  EtaPhiHists ChannelStatus;

}; //class HcalMonitorModule : public edm::EDAnalyzer

#endif
