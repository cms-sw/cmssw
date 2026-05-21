/*
   Implementation:
   This analyzer is copied from its AOD counterpart https://github.com/CmsHI/cmssw/blob/c3ad3155c9f57a40d2d81e488b1226cafa1722a5/HeavyIonsAnalysis/EventAnalysis/src/TriggerObjectAnalyzer.cc and adapted for TriggerObjectStandAlone object in miniAOD
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/HLTcore/interface/HLTConfigData.h"

#include "TNtuple.h"
#include "TRegexp.h"

//
// class declaration
//

class TriggerObjectAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  explicit TriggerObjectAnalyzer(const edm::ParameterSet&);
  ~TriggerObjectAnalyzer() override;

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

  std::string processName_;
  std::vector<std::string> triggerNames_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsTag_;
  edm::EDGetTokenT<std::vector<pat::TriggerObjectStandAlone>> triggerObjectsTag_;

  edm::Handle<edm::TriggerResults> triggerResultsHandle_;
  edm::Handle<std::vector<pat::TriggerObjectStandAlone>> triggerObjectsHandle_;

  HLTConfigProvider hltConfig_;

  unsigned int triggerIndex_;
  unsigned int moduleIndex_;
  std::string moduleLabel_;
  std::vector<std::string> moduleLabels_;

  edm::Service<TFileService> fs;
  std::vector<TTree*> nt_;
  int verbose_;

  std::map<std::string, bool> triggerInMenu;

  std::vector<double> id[500];
  std::vector<double> pt[500];
  std::vector<double> eta[500];
  std::vector<double> phi[500];
  std::vector<double> mass[500];
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
TriggerObjectAnalyzer::TriggerObjectAnalyzer(const edm::ParameterSet& ps) {
  processName_ = ps.getParameter<std::string>("processName");
  triggerNames_ = ps.getParameter<std::vector<std::string>>("triggerNames");
  triggerResultsTag_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("triggerResults"));
  triggerObjectsTag_ =
      consumes<std::vector<pat::TriggerObjectStandAlone>>(ps.getParameter<edm::InputTag>("triggerObjects"));

  //now do what ever initialization is needed
  nt_.reserve(triggerNames_.size());
  for (unsigned int itrig = 0; itrig < triggerNames_.size(); itrig++) {
    nt_[itrig] = fs->make<TTree>(triggerNames_.at(itrig).c_str(), Form("trigger %d", itrig));

    nt_[itrig]->Branch("TriggerObjID", &(id[itrig]));
    nt_[itrig]->Branch("pt", &(pt[itrig]));
    nt_[itrig]->Branch("eta", &(eta[itrig]));
    nt_[itrig]->Branch("phi", &(phi[itrig]));
    nt_[itrig]->Branch("mass", &(mass[itrig]));
  }

  verbose_ = 0;
}

TriggerObjectAnalyzer::~TriggerObjectAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void TriggerObjectAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (hltConfig_.size() > 0) {
    iEvent.getByToken(triggerResultsTag_, triggerResultsHandle_);
    iEvent.getByToken(triggerObjectsTag_, triggerObjectsHandle_);

    const edm::TriggerNames& triggerNamesEvt = iEvent.triggerNames(*triggerResultsHandle_);

    for (unsigned int itrig = 0; itrig < triggerNames_.size(); itrig++) {
      std::map<std::string, bool>::iterator inMenu = triggerInMenu.find(triggerNames_[itrig]);
      if (inMenu == triggerInMenu.end()) {
        continue;
      }

      triggerIndex_ = hltConfig_.triggerIndex(triggerNames_[itrig]);

      bool accepted = triggerResultsHandle_->accept(triggerIndex_);
      if (!accepted) {
        // do not fill trigger object if the event is not accepted.
        continue;
      }

      for (pat::TriggerObjectStandAlone tObj :
           *triggerObjectsHandle_) {  // note: not "const &" since we want to call unpackPathNames
        tObj.unpackPathNames(triggerNamesEvt);

        bool requireLastFilterAccept = true;
        bool requireL3FilterAccept = false;
        if (tObj.hasPathName(triggerNames_[itrig], requireLastFilterAccept, requireL3FilterAccept)) {
          if (verbose_) {
            std::cout << "verbose : " << tObj.pdgId() << " " << tObj.pt() << " " << tObj.et() << " " << tObj.eta()
                      << " " << tObj.phi() << " " << tObj.mass() << std::endl;
          }
          id[itrig].push_back(
              tObj.pdgId());  // TriggerObjectStandAlone::pdgId() does not necessarily correspond to what is in AOD counterpart, but its the closest by comparing descriptions. Also, this branch is rather useless, could be removed later on
          pt[itrig].push_back(tObj.pt());
          eta[itrig].push_back(tObj.eta());
          phi[itrig].push_back(tObj.phi());
          mass[itrig].push_back(tObj.mass());
        }
      }
    }
  }

  for (unsigned int itrig = 0; itrig < triggerNames_.size(); itrig++) {
    nt_[itrig]->Fill();
    id[itrig].clear();
    pt[itrig].clear();
    eta[itrig].clear();
    phi[itrig].clear();
    mass[itrig].clear();
  }
}

// ------------ method called once each job just before starting event loop  ------------
void TriggerObjectAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void TriggerObjectAnalyzer::endJob() {}

// ------------ method called when starting to process a run  ------------
void TriggerObjectAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed(true);
  if (hltConfig_.init(iRun, iSetup, processName_, changed)) {
    if (changed) {
      std::vector<std::string> activeHLTPathsInThisEvent = hltConfig_.triggerNames();

      triggerInMenu.clear();
      for (unsigned int itrig = 0; itrig < triggerNames_.size(); itrig++) {
        for (std::vector<std::string>::const_iterator iHLT = activeHLTPathsInThisEvent.begin();
             iHLT != activeHLTPathsInThisEvent.end();
             ++iHLT) {
          //matching with regexp filter name. More than 1 matching filter is allowed so trig versioning is transparent to analyzer
          if (TString(*iHLT).Contains(TRegexp(TString(triggerNames_[itrig])))) {
            triggerInMenu[*iHLT] = true;
            triggerNames_[itrig] = TString(*iHLT);
          }
        }
      }
      for (unsigned int itrig = 0; itrig < triggerNames_.size(); itrig++) {
        std::map<std::string, bool>::iterator inMenu = triggerInMenu.find(triggerNames_[itrig]);
        if (inMenu == triggerInMenu.end()) {
          std::cout << "<HLT Object Analyzer> Warning! Trigger " << triggerNames_[itrig]
                    << " not found in HLTMenu. Skipping..." << std::endl;
        }
      }
      if (verbose_) {
        hltConfig_.dump("ProcessName");
        hltConfig_.dump("GlobalTag");
        hltConfig_.dump("TableName");
        hltConfig_.dump("Streams");
        hltConfig_.dump("Datasets");
        hltConfig_.dump("PrescaleTable");
        hltConfig_.dump("ProcessPSet");
      }
    }
  } else {
    std::cout << "HLTObjectAnalyzer::analyze:"
              << " config extraction failure with process name " << processName_ << std::endl;
  }
}

// ------------ method called when ending the processing of a run  ------------
void TriggerObjectAnalyzer::endRun(edm::Run const&, edm::EventSetup const&) {}

// ------------ method called when starting to processes a luminosity block  ------------
void TriggerObjectAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

// ------------ method called when ending the processing of a luminosity block  ------------
void TriggerObjectAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TriggerObjectAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TriggerObjectAnalyzer);
