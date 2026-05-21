#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TUtmAlgorithm.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"

#include "TTree.h"

class TriggerAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  TriggerAnalyzer(edm::ParameterSet const& conf);
  ~TriggerAnalyzer() override;

  void analyze(const edm::Event& e, const edm::EventSetup& iSetup) override;
  void endJob() override;
  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
  void endRun(const edm::Run& run, const edm::EventSetup& es) override;

private:
  TTree* t_;

  unsigned long long fEvent;
  int fLumiBlock;
  int fRun;
  int fBx;
  int fOrbit;

  int HltEvtCnt;
  int L1EvtCnt;
  int* hltflag;
  int* l1flag;
  int* hltPrescaleNumerator;
  int* hltPrescaleDenominator;
  int* l1Prescl;

  std::string processName_;

  std::vector<std::string> hltdummies;
  std::vector<std::string> l1dummies;

  std::map<std::string, int> pathtoindex;

  edm::EDGetTokenT<edm::TriggerResults> hltresultsToken_;
  edm::EDGetTokenT<GlobalAlgBlkBxCollection> l1resultsToken_;
  edm::ESGetToken<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd> l1GtMenuToken_;

  std::unique_ptr<HLTPrescaleProvider> hltPrescaleProvider_;
};

static constexpr int kMaxHLTFlag = 1000;
static constexpr int kMaxL1Flag = 1000.;

TriggerAnalyzer::TriggerAnalyzer(edm::ParameterSet const& conf)
    : fEvent(0),
      fLumiBlock(-1),
      fRun(-1),
      fBx(-1),
      fOrbit(-1),
      HltEvtCnt(0),
      L1EvtCnt(0),
      hltflag(new int[kMaxHLTFlag]),
      l1flag(new int[kMaxL1Flag]),
      hltPrescaleNumerator(new int[kMaxHLTFlag]),
      hltPrescaleDenominator(new int[kMaxHLTFlag]),
      l1Prescl(new int[kMaxL1Flag]),
      processName_(conf.getParameter<std::string>("HLTProcessName")),
      hltdummies(conf.getParameter<std::vector<std::string>>("hltdummybranches")),
      l1dummies(conf.getParameter<std::vector<std::string>>("l1dummybranches")),
      hltresultsToken_(consumes<edm::TriggerResults>(conf.getParameter<edm::InputTag>("hltresults"))),
      l1resultsToken_(consumes<GlobalAlgBlkBxCollection>(conf.getParameter<edm::InputTag>("l1results"))),
      l1GtMenuToken_(esConsumes<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd>()),
      hltPrescaleProvider_(
          new HLTPrescaleProvider(conf.getParameter<edm::ParameterSet>("hltPSProvCfg"), consumesCollector(), *this)) {
  // open the tree file and initialize the tree
  edm::Service<TFileService> fs;
  t_ = fs->make<TTree>("HltTree", "");

  t_->Branch("Event", &fEvent, "Event/l");
  t_->Branch("LumiBlock", &fLumiBlock, "LumiBlock/I");
  t_->Branch("Run", &fRun, "Run/I");
  t_->Branch("Bx", &fBx, "Bx/I");
  t_->Branch("Orbit", &fOrbit, "Orbit/I");
}

TriggerAnalyzer::~TriggerAnalyzer() {
  delete[] hltflag;
  delete[] l1flag;
  delete[] hltPrescaleNumerator;
  delete[] hltPrescaleDenominator;
  delete[] l1Prescl;
}

void TriggerAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  edm::Handle<edm::TriggerResults> hltresults;
  edm::Handle<GlobalAlgBlkBxCollection> l1results;

  iEvent.getByToken(hltresultsToken_, hltresults);
  iEvent.getByToken(l1resultsToken_, l1results);

  fEvent = iEvent.id().event();
  fLumiBlock = iEvent.luminosityBlock();
  fRun = iEvent.id().run();
  fBx = iEvent.bunchCrossing();
  fOrbit = iEvent.orbitNumber();

  if (hltresults.isValid()) {
    /* reset accept status to -1 */
    for (int i = 0; i < kMaxHLTFlag; ++i) {
      hltflag[i] = -1;
      hltPrescaleNumerator[i] = -1;
      hltPrescaleDenominator[i] = -1;
    }

    int ntrigs = hltresults->size();
    if (ntrigs == 0) {
      edm::LogWarning("TriggerAnalyzer") << "-- No triggers found" << std::endl;
    }

    edm::TriggerNames const& triggerNames = iEvent.triggerNames(*hltresults);

    // 1st event : Book as many branches as trigger paths provided in the input...
    if (HltEvtCnt == 0) {
      int itdum = 0;
      for (auto const& dummy : hltdummies) {
        TString dummyname(dummy.data());
        t_->Branch(dummyname, hltflag + itdum, dummyname + "/I");
        t_->Branch(dummyname + "_PrescaleNumerator", hltPrescaleNumerator + itdum, dummyname + "_PrescaleNumerator/I");
        t_->Branch(dummyname + "_PrescaleDenominator", hltPrescaleDenominator + itdum, dummyname + "_PrescaleDenominator/I");
        pathtoindex[dummy] = itdum;
        ++itdum;
      }

      for (int itrig = 0; itrig != ntrigs; ++itrig) {
        const std::string& trigname = triggerNames.triggerName(itrig);
        if (pathtoindex.find(trigname) == pathtoindex.end()) {
          TString hltname = trigname;
          t_->Branch(hltname, hltflag + itdum + itrig, hltname + "/I");
          t_->Branch(hltname + "_PrescaleNumerator", hltPrescaleNumerator + itdum + itrig, hltname + "_PrescaleNumerator/I");
          t_->Branch(hltname + "_PrescaleDenominator", hltPrescaleDenominator + itdum + itrig, hltname + "_PrescaleDenominator/I");
          pathtoindex[trigname] = itdum + itrig;
        }
      }

      HltEvtCnt++;
    }
    // ...Fill the corresponding accepts in branch-variables

    FractionalPrescale thisTriggerPrescale;
    for (int itrig = 0; itrig != ntrigs; ++itrig) {
      const std::string& trigname = triggerNames.triggerName(itrig);
      bool accept = hltresults->accept(itrig);

      int index = pathtoindex[trigname];
      thisTriggerPrescale = hltPrescaleProvider_->prescaleValue<FractionalPrescale>(iEvent, iSetup, trigname);
      hltPrescaleNumerator[index] = thisTriggerPrescale.numerator();
      hltPrescaleDenominator[index] = thisTriggerPrescale.denominator();

      hltflag[index] = accept;
    }
  } else {
    edm::LogInfo("TriggerAnalyzer") << "-- No Trigger Result" << std::endl;
  }

  auto& l1GtUtils = const_cast<l1t::L1TGlobalUtil&>(hltPrescaleProvider_->l1tGlobalUtil());

  l1GtUtils.retrieveL1(iEvent, iSetup);

  //edm::ESHandle<L1TUtmTriggerMenu> menu;
  //auto const& menu = iSetup.getData(l1GtMenuToken_);
  auto const& menu = iSetup.getHandle(l1GtMenuToken_);
  //iSetup.get<L1TUtmTriggerMenuRcd>().get(menu);

  if (l1results.isValid() && l1results->size() != 0) {
    /* reset accept status to -1 */
    for (int i = 0; i < kMaxL1Flag; ++i) {
      l1flag[i] = -1;
      l1Prescl[i] = -1;
    }

    // 1st event : Book as many branches as trigger paths provided in the input...
    if (L1EvtCnt == 0) {
      int itdum = 0;
      for (auto const& dummy : l1dummies) {
        TString dummyname(dummy.data());
        t_->Branch(dummyname, l1flag + itdum, dummyname + "/I");
        t_->Branch(dummyname + "_Prescl", l1Prescl + itdum, dummyname + "_Prescl/I");
        pathtoindex[dummy] = itdum;
        ++itdum;
      }

      int il1 = 0;
      // get the bit/name association
      for (auto const& keyval : menu->getAlgorithmMap()) {
        std::string const& trigname = keyval.second.getName();

        if (pathtoindex.find(trigname) == pathtoindex.end()) {
          TString l1name = trigname;
          t_->Branch(l1name, l1flag + itdum + il1, l1name + "/I");
          t_->Branch(l1name + "_Prescl", l1Prescl + itdum + il1, l1name + "_Prescl/I");
          pathtoindex[trigname] = itdum + il1;
          ++il1;
        }
      }  // end algo Map

      L1EvtCnt++;
    }  // end l1evtCnt=0

    GlobalAlgBlk const& result = l1results->at(0, 0);

    // get the individual decisions from the GlobalAlgBlk
    for (auto const& keyval : menu->getAlgorithmMap()) {
      auto const& l1name = keyval.second.getName();
      int l1index = keyval.second.getIndex();

      int index = pathtoindex[l1name];

      bool accept = result.getAlgoDecisionFinal(l1index);
      l1flag[index] = accept;

      double prescale = -1;
      l1GtUtils.getPrescaleByBit(l1index, prescale);
      l1Prescl[index] = prescale;
    }

    // l1results.isValid
  } else {
    edm::LogWarning("TriggerAnalyzer") << "%L1Results -- No L1 Results" << std::endl;
  }

  // After analysis, fill the variables tree
  t_->Fill();
}

// ------------ method called when starting to process a run  ------------
void TriggerAnalyzer::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  bool changed(true);
  if (hltPrescaleProvider_->init(run, es, processName_, changed)) {
    // if init returns TRUE, initialisation has succeeded!
    if (changed) {
      edm::LogWarning("TriggerAnalyzer") << "HLT config change" << std::endl;
    }
  } else {
    // if init returns FALSE, initialisation has NOT succeeded, which indicates
    // a problem with the file and/or code and needs to be investigated!
    edm::LogInfo("TriggerAnalyzer") << " HLT initialisation failed for process " << processName_ << std::endl;
    // In this case, all access methods will return empty values!
  }
}

// ------------ method called when processing run ends  ------------
void TriggerAnalyzer::endRun(const edm::Run& run, const edm::EventSetup& es) {}

void TriggerAnalyzer::endJob() {}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerAnalyzer);
