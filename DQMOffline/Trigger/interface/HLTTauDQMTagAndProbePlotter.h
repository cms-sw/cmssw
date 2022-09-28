// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMTagAndProbePlotter_h
#define DQMOffline_Trigger_HLTTauDQMTagAndProbePlotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMPath.h"
#include "DQMOffline/Trigger/interface/HistoWrapper.h"

//#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "FWCore/Framework/interface/Event.h"

#include "Math/GenVector/VectorUtil.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

namespace trigger {
  class TriggerEvent;
}

class HLTConfigProvider;

class HLTTauDQMTagAndProbePlotter : private HLTTauDQMPlotter {
public:
  HLTTauDQMTagAndProbePlotter(const edm::ParameterSet &iConfig,
                              const std::vector<std::string> &modLabels,
                              const std::string &dqmBaseFolder);
  ~HLTTauDQMTagAndProbePlotter();

  using HLTTauDQMPlotter::isValid;

  void bookHistograms(HistoWrapper &iWrapper,
                      DQMStore::IBooker &iBooker,
                      edm::Run const &iRun,
                      edm::EventSetup const &iSetup);
  template <class T>
  void analyze(edm::Event const &iEvent,
               const edm::TriggerResults &triggerResults,
               const T &triggerEvent,
               const HLTTauDQMOfflineObjects &refCollection) {
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

    const edm::TriggerNames &trigNames = iEvent.triggerNames(triggerResults);

    for (const LV &offlineObject : offlineObjects) {
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

      if (h_den_pt)
        h_den_pt->Fill(offlineObject.pt());
      if (xvariable != "met") {
        if (h_den_eta)
          h_den_eta->Fill(offlineObject.eta());
        if (h_den_etaphi)
          h_den_etaphi->Fill(offlineObject.eta(), offlineObject.phi());
      }
      if (h_den_phi)
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
      if (h_num_pt)
        h_num_pt->Fill(offlineObject.pt());
      if (xvariable != "met") {
        if (h_num_eta)
          h_num_eta->Fill(offlineObject.eta());
        if (h_num_etaphi)
          h_num_etaphi->Fill(offlineObject.eta(), offlineObject.phi());
      }
      if (h_num_phi)
        h_num_phi->Fill(offlineObject.phi());
    }
  }

private:
  LV findTrgObject(std::string, const trigger::TriggerEvent &);
  LV findTrgObject(std::string, const pat::TriggerObjectStandAloneCollection &);

  const int nbinsPt_;
  const double ptmin_, ptmax_;
  int nbinsEta_;
  double etamin_, etamax_;
  const int nbinsPhi_;
  const double phimin_, phimax_;
  std::string xvariable;

  std::vector<std::string> numTriggers;
  std::vector<std::string> denTriggers;

  std::vector<std::string> moduleLabels;

  unsigned int nOfflineObjs;

  MonitorElement *h_num_pt;
  MonitorElement *h_den_pt;

  MonitorElement *h_num_eta;
  MonitorElement *h_den_eta;

  MonitorElement *h_num_phi;
  MonitorElement *h_den_phi;

  MonitorElement *h_num_etaphi;
  MonitorElement *h_den_etaphi;
};

#endif
