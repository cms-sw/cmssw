#ifndef DQMPROVINFO_H
#define DQMPROVINFO_H

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Run.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/ServiceRegistry/interface/Service.h>

#include <DQMServices/Core/interface/DQMOneEDAnalyzer.h>
#include <DQMServices/Core/interface/DQMStore.h>

#include <DataFormats/Scalers/interface/DcsStatus.h>

#include <DataFormats/TCDS/interface/TCDSRecord.h>
#include <DataFormats/OnlineMetaData/interface/DCSRecord.h>

#include <string>
#include <vector>

class DQMProvInfo : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  // Constructor
  DQMProvInfo(const edm::ParameterSet& ps);
  // Destructor
  ~DQMProvInfo() override;

protected:
  void dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c) override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c) override;

private:
  void bookHistogramsLhcInfo(DQMStore::IBooker&);
  void bookHistogramsEventInfo(DQMStore::IBooker&);
  void bookHistogramsProvInfo(DQMStore::IBooker&);

  void analyzeLhcInfo(const edm::Event& e);
  void analyzeEventInfo(const edm::Event& e);
  void analyzeProvInfo(const edm::Event& e);

  void fillDcsBitsFromDCSRecord(const DCSRecord&);
  void fillDcsBitsFromDcsStatusCollection(const edm::Handle<DcsStatusCollection>&);
  bool isPhysicsDeclared();

  void endLuminosityBlockLhcInfo(const int currentLSNumber);
  void endLuminosityBlockEventInfo(const int currentLSNumber);
  void blankPreviousLumiSections(const int currentLSNumber);
  void blankAllLumiSections();

  // To max amount of lumisections we foresee for the plots
  // DQM GUI renderplugins provide scaling to actual amount
  const static int MAX_LUMIS = 6000;

  // Numbers of each of the vertical bins
  const static int VBIN_CSC_P = 1;
  const static int VBIN_CSC_M = 2;
  const static int VBIN_DT_0 = 3;
  const static int VBIN_DT_P = 4;
  const static int VBIN_DT_M = 5;
  const static int VBIN_EB_P = 6;
  const static int VBIN_EB_M = 7;
  const static int VBIN_EE_P = 8;
  const static int VBIN_EE_M = 9;
  const static int VBIN_ES_P = 10;
  const static int VBIN_ES_M = 11;
  const static int VBIN_HBHE_A = 12;
  const static int VBIN_HBHE_B = 13;
  const static int VBIN_HBHE_C = 14;
  const static int VBIN_HF = 15;
  const static int VBIN_HO = 16;
  const static int VBIN_BPIX = 17;
  const static int VBIN_FPIX = 18;
  const static int VBIN_RPC = 19;
  const static int VBIN_TIBTID = 20;
  const static int VBIN_TOB = 21;
  const static int VBIN_TEC_P = 22;
  const static int VBIN_TE_M = 23;
  const static int VBIN_CASTOR = 24;
  const static int VBIN_ZDC = 25;

  // Highest DCS bin, used for the length of the corresponding array.
  // We will have the indexes to this array the same as the vbins numbers.
  // (I.e. value at index 0 will not be used.)
  const static int MAX_DCS_VBINS = 25;

  const static int VBIN_PHYSICS_DECLARED = 26;
  const static int VBIN_MOMENTUM = 27;
  const static int VBIN_STABLE_BEAM = 28;
  const static int VBIN_VALID = 29;

  const static int MAX_VBINS = 29;

  // Beam momentum at flat top, used to determine if collisions are
  // occurring with the beams at the energy allowed for physics production.
  const static int MAX_MOMENTUM = 6500;

  // Beam momentum allowed offset: it is a momentum value subtracted to
  // maximum momentum in order to decrease the threshold for beams going to
  // collisions for physics production. This happens because BST sends from
  // time to time a value of the beam momentum slightly below the nominal values,
  // even during stable collisions: in this way, we provide a correct information
  // at the cost of not requiring the exact momentum being measured by BST.
  const static int MOMENTUM_OFFSET = 1;

  // Process parameters
  std::string subsystemname_;
  std::string provinfofolder_;

  edm::EDGetTokenT<DcsStatusCollection> dcsStatusCollection_;
  edm::EDGetTokenT<TCDSRecord> tcdsrecord_;
  edm::EDGetTokenT<DCSRecord> dcsRecordToken_;

  // MonitorElements for LhcInfo and corresponding variables
  MonitorElement* hBeamMode_;
  int beamMode_;
  MonitorElement* hIntensity1_;
  int intensity1_;
  MonitorElement* hIntensity2_;
  int intensity2_;
  MonitorElement* hLhcFill_;
  int lhcFill_;
  MonitorElement* hMomentum_;
  int momentum_;

  // MonitorElements for EventInfo and corresponding variables
  MonitorElement* reportSummary_;
  MonitorElement* reportSummaryMap_;
  int previousLSNumber_;
  bool physicsDeclared_;
  bool foundFirstPhysicsDeclared_;
  bool dcsBits_[MAX_DCS_VBINS + 1];
  bool foundFirstDcsBits_;

  // MonitorElements for ProvInfo and corresponding variables
  MonitorElement* versCMSSW_;
  MonitorElement* versGlobaltag_;
  std::string globalTag_;
  bool globalTagRetrieved_;
  MonitorElement* versRuntype_;
  std::string runType_;
  MonitorElement* hHltKey_;
  std::string hltKey_;
  MonitorElement* hostName_;
  MonitorElement* hIsCollisionsRun_;
  MonitorElement* processId_;  // The PID associated with this job
  MonitorElement* workingDir_;
};

#endif
