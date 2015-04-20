#ifndef HcalMonitorModule_GUARD_H
#define HcalMonitorModule_GUARD_H

/*
 * \file HcalMonitorModule.h
 *

 * \author J. Temple
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
// forward declarations

class MonitorElement;
class FEDRawDataCollection;
class HcalElectronicsMap;

class HcalMonitorModule : public DQMEDAnalyzer
{

public:

  // Constructor
  HcalMonitorModule(const edm::ParameterSet& ps);

  // Destructor
  ~HcalMonitorModule();

  void dqmBeginRun(edm::Run const &, edm::EventSetup const &);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);

 protected:

  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  // End LumiBlock
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                         const edm::EventSetup& c);

  // EndRun
  void endRun(const edm::Run& run, const edm::EventSetup& c);

  // Reset
  void reset(void);

  // cleanup
  void cleanup(void);

  // setup
  void setup(DQMStore::IBooker &);
  
  // CheckSubdetectorStatus
  void CheckSubdetectorStatus(const edm::Handle<FEDRawDataCollection>& rawraw,
			      HcalSubdetector subdet,
			      const HcalElectronicsMap& emap);

 private:

  int ievt_;
  edm::RunNumber_t runNumber_;
  edm::EventNumber_t evtNumber_;

  MonitorElement* meStatus_;
  MonitorElement* meRun_;
  MonitorElement* meEvt_;
  MonitorElement* meFEDS_;
  MonitorElement *meUTCAFEDS_;
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
  edm::InputTag FEDRawDataCollection_;
  edm::InputTag inputLabelReport_;
//  edm::InputTag inputLabelReportUTCA_;
  std::string prefixME_;
  int NLumiBlocks_;

  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
  edm::EDGetTokenT<HcalUnpackerReport> tok_report_;
//  edm::EDGetTokenT<HcalUnpackerReport> tok_reportUTCA_;

  int HBpresent_, HEpresent_, HOpresent_, HFpresent_;

  const HcalElectronicsMap*    eMap_;
  EtaPhiHists ChannelStatus;
//  std::vector<int> _feds;

}; //class HcalMonitorModule : public edm::EDAnalyzer

#endif
