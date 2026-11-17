#include <algorithm>
#include <utility>
#include "FWCore/Utilities/interface/Exception.h"

// system include files
#include <memory>
#include <iostream>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TTree.h"

using namespace std;

//
// class declaration
//

class FilterAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  explicit FilterAnalyzer(const edm::ParameterSet&);
  ~FilterAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::TriggerResults> hltresults_;
  vector<string> superFilters_;
  bool _Debug;

  int HltEvtCnt;

  TTree* HltTree;
  std::vector<std::string> branchesToKeep_;
  int* trigflag;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FilterAnalyzer::FilterAnalyzer(const edm::ParameterSet& conf) {
  //now do what ever initialization is needed
  hltresults_ = consumes<edm::TriggerResults>(conf.getParameter<edm::InputTag>("hltresults"));
  superFilters_ = conf.getParameter<vector<string> >("superFilters");
  _Debug = conf.getUntrackedParameter<bool>("Debug", false);

  HltEvtCnt = 0;
  edm::Service<TFileService> fs;
  HltTree = fs->make<TTree>("HltTree", "");
  branchesToKeep_ = conf.getUntrackedParameter<std::vector<std::string>>("branchesToKeep", {});

  const int kMaxTrigFlag = 10000;
  trigflag = new int[kMaxTrigFlag];
}

FilterAnalyzer::~FilterAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void FilterAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto bookBranch = [this](const char* name, auto&&... args) {
    if (branchesToKeep_.empty() ||
        std::find(branchesToKeep_.begin(), branchesToKeep_.end(), name) != branchesToKeep_.end())
      HltTree->Branch(name, std::forward<decltype(args)>(args)...);
  };

  using namespace edm;
  edm::Handle<edm::TriggerResults> hltresults;

  iEvent.getByToken(hltresults_, hltresults);
  edm::TriggerNames const& triggerNames = iEvent.triggerNames(*hltresults);
  int ntrigs = hltresults->size();

  if (HltEvtCnt == 0) {
    for (int itrig = 0; itrig != ntrigs; ++itrig) {
      TString trigName = triggerNames.triggerName(itrig);
      bookBranch(trigName, trigflag + itrig, trigName + "/I");
    }

      for (const auto& name : branchesToKeep_)
        if (!HltTree->GetBranch(name.c_str()))
          throw cms::Exception("Configuration") << "branchesToKeep requested an unavailable branch: " << name;
    HltEvtCnt++;
  }
  bool saveEvent = true;

  for (int itrig = 0; itrig != ntrigs; ++itrig) {
    std::string trigName = triggerNames.triggerName(itrig);
    bool accept = hltresults->accept(itrig);

    for (unsigned int ifilter = 0; ifilter < superFilters_.size(); ++ifilter) {
      if (_Debug)
        cout << "trigName " << trigName.data() << "    superFilters_[ifilter] " << superFilters_[ifilter] << endl;
      if (trigName == superFilters_[ifilter])
        saveEvent = saveEvent && accept;
    }

    if (accept) {
      trigflag[itrig] = 1;
    } else {
      trigflag[itrig] = 0;
    }

    if (_Debug) {
      if (_Debug)
        std::cout << "%HLTInfo --  Number of HLT Triggers: " << ntrigs << std::endl;
      std::cout << "%HLTInfo --  HLTTrigger(" << itrig << "): " << trigName << " = " << accept << std::endl;
    }
  }
  if (saveEvent)
    HltTree->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void FilterAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void FilterAnalyzer::endJob() {}

// ------------ method called when starting to processes a run  ------------
void FilterAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&) {}

// ------------ method called when ending the processing of a run  ------------
void FilterAnalyzer::endRun(edm::Run const&, edm::EventSetup const&) {}

// ------------ method called when starting to processes a luminosity block  ------------
void FilterAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

// ------------ method called when ending the processing of a luminosity block  ------------
void FilterAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void FilterAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(FilterAnalyzer);
