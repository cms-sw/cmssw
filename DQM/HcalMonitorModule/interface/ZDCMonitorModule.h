#ifndef ZDCMonitorModule_GUARD_H
#define ZDCMonitorModule_GUARD_H

/*
 * \file ZDCMonitorModule.h
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

class ZDCMonitorModule : public edm::EDAnalyzer
{

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

  // cleanup
  void cleanup(void);

  // setup
  void setup(void);
  
  // CheckSubdetectorStatus
  void CheckSubdetectorStatus(const edm::Handle<FEDRawDataCollection>& rawraw,
			      const HcalElectronicsMap& emap);

 private:

  int ievt_;
  int runNumber_;
  int evtNumber_;

  // Not sure how many of these are needed for ZDC

  MonitorElement* meStatus_;
  MonitorElement* meRun_;
  MonitorElement* meEvt_;
  MonitorElement* meFEDS_;
  MonitorElement* meCalibType_;
  MonitorElement* meCurrentCalibType_;
  MonitorElement* meZDC_;
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

  int ZDCpresent_;
  DQMStore* dbe_;

  const HcalElectronicsMap*    eMap_;
  //EtaPhiHists ChannelStatus;  // general Hcal Eta-Phi histograms not needed for ZDC Monitor Module.  Replace with ZDC-specific histogram?

}; //class ZDCMonitorModule : public edm::EDAnalyzer

#endif
