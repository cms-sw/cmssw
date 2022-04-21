// user includes
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "DQM/HLTEvF/interface/PathTimerService.h"

// ROOT includes
#include "TFile.h"
#include "TTree.h"

class SiStripModuleTimer : public edm::EDAnalyzer {
public:
  SiStripModuleTimer(const edm::ParameterSet&);
  ~SiStripModuleTimer();

  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();

private:
  std::vector<std::string> moduleLabels_;
  std::vector<double> times_;
  TFile* file_;
  TTree* tree_;
};

using namespace std;
using namespace edm;

SiStripModuleTimer::SiStripModuleTimer(const ParameterSet& pset)
    : moduleLabels_(pset.getUntrackedParameter<vector<string> >("ModuleLabels")),
      times_(moduleLabels_.size()),
      file_(0),
      tree_(0) {
  file_ = new TFile(pset.getUntrackedParameter<string>("FileName", "SiStripTiming.root").c_str(), "UPDATE");
  tree_ = new TTree(pset.getUntrackedParameter<string>("TreeName", "Tree").c_str(), "");
  for (unsigned short i = 0; i < moduleLabels_.size(); i++) {
    std::string label = moduleLabels_[i];
    std::string type = moduleLabels_[i] + "/D";
    tree_->Branch(label.c_str(), &times_[i], type.c_str());
  }
}

SiStripModuleTimer::~SiStripModuleTimer() {
  if (tree_)
    delete tree_;
  file_->Close();
}

void SiStripModuleTimer::endJob() {
  file_->cd();
  tree_->Write();
}

void SiStripModuleTimer::analyze(const Event& iEvent, const EventSetup& iSetup) {
  times_.assign(moduleLabels_.size(), 0.);
  //auto_ptr<HLTPerformanceInfo> hltinfo = Service<service::PathTimerService>().operator->()->getInfo();
  //HLTPerformanceInfo::Modules::const_iterator imodule = hltinfo->beginModules();
  //for (;imodule != hltinfo->endModules(); imodule++) {
  //  for (unsigned short i=0; i<moduleLabels_.size(); i++) {
  //    if (imodule->name() == moduleLabels_[i]) times_[i]=imodule->time();
  //  }
  //}
  tree_->Fill();
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripModuleTimer);
