#ifndef SUSY_HLT_PhotonMET_H
#define SUSY_HLT_PhotonMET_H

// event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

// DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

// MET
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

// Photon
#include "DataFormats/EgammaCandidates/interface/Photon.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"

namespace reco {
  typedef std::vector<reco::Photon> PhotonCollection;
}

class SUSY_HLT_PhotonMET : public DQMEDAnalyzer {
public:
  SUSY_HLT_PhotonMET(const edm::ParameterSet &ps);
  ~SUSY_HLT_PhotonMET() override;

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;

private:
  // histos booking function
  void bookHistos(DQMStore::IBooker &);

  // variables from config file
  edm::EDGetTokenT<reco::PFMETCollection> thePfMETCollection_;
  edm::EDGetTokenT<reco::PhotonCollection> thePhotonCollection_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResults_;

  std::string triggerPath_;
  std::string triggerPathBase_;
  edm::InputTag triggerFilterPhoton_;
  edm::InputTag triggerFilterMET_;
  double ptThrOffline_;
  double metThrOffline_;

  // Histograms
  MonitorElement *h_recoPhotonPt;
  MonitorElement *h_recoMet;
  MonitorElement *h_metTurnOn_num;
  MonitorElement *h_metTurnOn_den;
  MonitorElement *h_photonTurnOn_num;
  MonitorElement *h_photonTurnOn_den;
};

#endif
