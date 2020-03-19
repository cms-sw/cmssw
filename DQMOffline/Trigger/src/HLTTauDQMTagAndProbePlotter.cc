#include "DQMOffline/Trigger/interface/HLTTauDQMTagAndProbePlotter.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include <boost/algorithm/string.hpp>
#include "Math/GenVector/VectorUtil.h"

namespace {
  std::string stripVersion(const std::string& pathName) {
    size_t versionStart = pathName.rfind("_v");
    if (versionStart == std::string::npos)
      return pathName;
    return pathName.substr(0, versionStart);
  }
}  // namespace

//HLTTauDQMTagAndProbePlotter::HLTTauDQMTagAndProbePlotter(const edm::ParameterSet& iConfig, std::unique_ptr<GenericTriggerEventFlag> numFlag, std::unique_ptr<GenericTriggerEventFlag> denFlag, const std::string& dqmBaseFolder) :
HLTTauDQMTagAndProbePlotter::HLTTauDQMTagAndProbePlotter(const edm::ParameterSet& iConfig,
                                                         const std::vector<std::string>& modLabels,
                                                         const std::string& dqmBaseFolder)
    : HLTTauDQMPlotter(stripVersion(iConfig.getParameter<std::string>("name")), dqmBaseFolder),
      nbinsPt_(iConfig.getParameter<int>("nPtBins")),
      ptmin_(iConfig.getParameter<double>("ptmin")),
      ptmax_(iConfig.getParameter<double>("ptmax")),
      nbinsPhi_(iConfig.getParameter<int>("nPhiBins")),
      phimin_(iConfig.getParameter<double>("phimin")),
      phimax_(iConfig.getParameter<double>("phimax")),
      xvariable(iConfig.getParameter<std::string>("xvariable")) {
  numTriggers =
      iConfig.getParameter<edm::ParameterSet>("numerator").getParameter<std::vector<std::string> >("hltPaths");
  denTriggers =
      iConfig.getParameter<edm::ParameterSet>("denominator").getParameter<std::vector<std::string> >("hltPaths");

  moduleLabels = modLabels;

  boost::algorithm::to_lower(xvariable);

  if (xvariable != "met") {
    nbinsEta_ = iConfig.getParameter<int>("nEtaBins");
    etamin_ = iConfig.getParameter<double>("etamin");
    etamax_ = iConfig.getParameter<double>("etamax");
  }

  nOfflineObjs = iConfig.getUntrackedParameter<unsigned int>("nOfflObjs", 1);
}

#include <algorithm>
void HLTTauDQMTagAndProbePlotter::bookHistograms(DQMStore::IBooker& iBooker,
                                                 edm::Run const& iRun,
                                                 edm::EventSetup const& iSetup) {
  if (!isValid())
    return;

  // Efficiency helpers
  iBooker.setCurrentFolder(triggerTag() + "/helpers");
  h_num_pt = iBooker.book1D(xvariable + "EtEffNum", "", nbinsPt_, ptmin_, ptmax_);
  h_den_pt = iBooker.book1D(xvariable + "EtEffDenom", "", nbinsPt_, ptmin_, ptmax_);

  if (xvariable != "met") {
    h_num_eta = iBooker.book1D(xvariable + "EtaEffNum", "", nbinsEta_, etamin_, etamax_);
    h_den_eta = iBooker.book1D(xvariable + "EtaEffDenom", "", nbinsEta_, etamin_, etamax_);

    h_num_etaphi =
        iBooker.book2D(xvariable + "EtaPhiEffNum", "", nbinsEta_, etamin_, etamax_, nbinsPhi_, phimin_, phimax_);
    h_den_etaphi =
        iBooker.book2D(xvariable + "EtaPhiEffDenom", "", nbinsEta_, etamin_, etamax_, nbinsPhi_, phimin_, phimax_);
    h_den_etaphi->setOption("COL");
  }

  h_num_phi = iBooker.book1D(xvariable + "PhiEffNum", "", nbinsPhi_, phimin_, phimax_);
  h_den_phi = iBooker.book1D(xvariable + "PhiEffDenom", "", nbinsPhi_, phimin_, phimax_);

  iBooker.setCurrentFolder(triggerTag());
}

HLTTauDQMTagAndProbePlotter::~HLTTauDQMTagAndProbePlotter() = default;

LV HLTTauDQMTagAndProbePlotter::findTrgObject(std::string pathName, const trigger::TriggerEvent& triggerEvent) {
  trigger::TriggerObjectCollection trigObjs = triggerEvent.getObjects();
  const unsigned moduleIndex = moduleLabels.size() - 2;

  const unsigned hltFilterIndex = triggerEvent.filterIndex(edm::InputTag(moduleLabels[moduleIndex], "", "HLT"));

  if (hltFilterIndex < triggerEvent.sizeFilters()) {
    const trigger::Keys& triggerKeys(triggerEvent.filterKeys(hltFilterIndex));
    const trigger::Vids& triggerVids(triggerEvent.filterIds(hltFilterIndex));

    const unsigned nTriggers = triggerVids.size();
    for (size_t iTrig = 0; iTrig < nTriggers; ++iTrig) {
      const trigger::TriggerObject trigObject = trigObjs[triggerKeys[iTrig]];
      //         std::cout << "        trigger objs pt,eta,phi: " << triggerKeys[iTrig] << " "
      //                   << trigObject.pt() << " " << trigObject.eta() << " " << trigObject.phi() << " " << trigObject.id() << std::endl;
      return LV(trigObject.px(), trigObject.py(), trigObject.pz(), trigObject.energy());
    }
  }
  return LV(0, 0, 0, 0);
}

void HLTTauDQMTagAndProbePlotter::analyze(edm::Event const& iEvent,
                                          const edm::TriggerResults& triggerResults,
                                          const trigger::TriggerEvent& triggerEvent,
                                          const HLTTauDQMOfflineObjects& refCollection) {
  std::vector<LV> offlineObjects;
  if (xvariable == "tau")
    offlineObjects = refCollection.taus;
  if (xvariable == "muon")
    offlineObjects = refCollection.muons;
  if (xvariable == "electron")
    offlineObjects = refCollection.electrons;
  if (xvariable == "met")
    offlineObjects = refCollection.met;

  if (offlineObjects.size() < nOfflineObjs)
    return;

  const edm::TriggerNames& trigNames = iEvent.triggerNames(triggerResults);

  for (const LV& offlineObject : offlineObjects) {
    // Filter out events if Trigger Filtering is requested
    bool passTrigger = false;
    bool hltMatched = false;
    for (size_t i = 0; i < denTriggers.size(); ++i) {
      LV trgObject = findTrgObject(denTriggers[i], triggerEvent);

      for (unsigned int hltIndex = 0; hltIndex < trigNames.size(); ++hltIndex) {
        passTrigger = (trigNames.triggerName(hltIndex).find(denTriggers[i]) != std::string::npos &&
                       triggerResults.wasrun(hltIndex) && triggerResults.accept(hltIndex));

        if (passTrigger) {
          double dr = ROOT::Math::VectorUtil::DeltaR(trgObject, offlineObject);
          if (dr < 0.4)
            hltMatched = true;
          break;
        }
      }
      if (passTrigger)
        break;
    }
    if (!passTrigger)
      return;
    if (hltMatched)
      return;  // do not consider offline objects which match the tag trigger

    h_den_pt->Fill(offlineObject.pt());
    if (xvariable != "met") {
      h_den_eta->Fill(offlineObject.eta());
      h_den_etaphi->Fill(offlineObject.eta(), offlineObject.phi());
    }
    h_den_phi->Fill(offlineObject.phi());

    // applying selection for numerator
    passTrigger = false;
    for (size_t i = 0; i < numTriggers.size(); ++i) {
      for (unsigned int hltIndex = 0; hltIndex < trigNames.size(); ++hltIndex) {
        passTrigger = (trigNames.triggerName(hltIndex).find(numTriggers[i]) != std::string::npos &&
                       triggerResults.wasrun(hltIndex) && triggerResults.accept(hltIndex));
        if (passTrigger)
          break;
      }
      if (passTrigger)
        break;
    }
    if (!passTrigger)
      return;

    h_num_pt->Fill(offlineObject.pt());
    if (xvariable != "met") {
      h_num_eta->Fill(offlineObject.eta());
      h_num_etaphi->Fill(offlineObject.eta(), offlineObject.phi());
    }
    h_num_phi->Fill(offlineObject.phi());
  }
}
