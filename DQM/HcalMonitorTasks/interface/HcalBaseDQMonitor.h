#ifndef DQM_HCALMONITORTASKS_GUARD_HCALBASE_H
#define DQM_HCALMONITORTASKS_GUARD_HCALBASE_H

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h" // needed to grab objects

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

class HcalLogicalMap;
class HcalElectronicsMap;

class HcalBaseDQMonitor : public DQMEDAnalyzer
{

public:

  // Constructor
  HcalBaseDQMonitor(const edm::ParameterSet& ps);
  // Constructor with no arguments
 HcalBaseDQMonitor():logicalMap_(0),needLogicalMap_(false),setupDone_(false){};

  // Destructor
  virtual ~HcalBaseDQMonitor();

  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);

protected:

  // Analyze
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  void getLogicalMap(const edm::EventSetup& c);
 
  // BeginRun
  virtual void dqmBeginRun(const edm::Run& run, const edm::EventSetup& c);

  // Begin LumiBlock
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                            const edm::EventSetup& c) ;

  // End LumiBlock
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                          const edm::EventSetup& c);

  // EndRun
  virtual void endRun(const edm::Run& run, const edm::EventSetup& c);

  // Reset
  virtual void reset(void);

  // cleanup
  virtual void cleanup(void);

  // setup
  virtual void setup(DQMStore::IBooker &);
  
  // LumiOutOfOrder
  bool LumiInOrder(int lumisec);

  void SetupEtaPhiHists(DQMStore::IBooker &ib, EtaPhiHists & hh, std::string Name, std::string Units)
  {
    hh.setup(ib, Name, Units);
    return;
  }

  // IsAllowedCalibType
  bool IsAllowedCalibType();
  int currenttype_;

  std::vector<int> AllowedCalibTypes_;
  bool Online_;
  bool mergeRuns_;
  bool enableCleanup_;
  int debug_;
  std::string prefixME_;
  std::string subdir_;

  int currentLS;
  int ievt_;
  int levt_; // number of events in current lumi block
  int tevt_; // number of events overall
  MonitorElement* meIevt_;
  MonitorElement* meTevt_;
  MonitorElement* meLevt_;
  MonitorElement* meTevtHist_;

  bool eventAllowed_;
  bool skipOutOfOrderLS_;
  bool makeDiagnostics_;

  // check that each subdetector is present
  bool HBpresent_, HEpresent_, HOpresent_, HFpresent_;

  // Define problem-tracking monitor elements -- keep here, or in the client?
  MonitorElement *ProblemsVsLB;
  MonitorElement *ProblemsVsLB_HB, *ProblemsVsLB_HE;
  MonitorElement *ProblemsVsLB_HO, *ProblemsVsLB_HF, *ProblemsVsLB_HBHEHF;
  MonitorElement* ProblemsCurrentLB;  // show problems just for this LB
 
  int NLumiBlocks_;
  // Store known channels to be ignored during plots of problems vs LB
  // store vector of vectors
  // index 0 = HB, 1 = HE, 2 = HF, 3 HO  (index = subdetector - 1)
  std::map<unsigned int, int> KnownBadCells_;

  HcalLogicalMap* logicalMap_;
  bool needLogicalMap_;

  int badChannelStatusMask_;


  private:
  bool setupDone_;

  // methods to check for sub-detector status
  void CheckSubdetectorStatus(const edm::Handle<FEDRawDataCollection>&, HcalSubdetector, const HcalElectronicsMap &);
  void CheckCalibType(const edm::Handle<FEDRawDataCollection>&); 

  edm::InputTag FEDRawDataCollection_;
  edm::EDGetTokenT<FEDRawDataCollection> tok_braw_;

  const HcalElectronicsMap * eMap_;
  
};// class HcalBaseDQMonitor : public edm::EDAnalyzer


#endif
