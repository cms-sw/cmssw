#ifndef SUSY_HLT_PhotonHT_H
#define SUSY_HLT_PhotonHT_H

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

class SUSY_HLT_PhotonHT : public DQMEDAnalyzer {
public:
  SUSY_HLT_PhotonHT(const edm::ParameterSet &ps);
  ~SUSY_HLT_PhotonHT() override;

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
  edm::EDGetTokenT<trigger::TriggerEvent> theTrigSummary_;

  std::string triggerPath_;
  std::string triggerPathAuxiliaryForHadronic_;
  edm::InputTag triggerFilterPhoton_;
  edm::InputTag triggerFilterHt_;
  double ptThrOffline_;
  double htThrOffline_;

  // Histograms
  MonitorElement *h_photonPt;
  MonitorElement *h_ht;
  MonitorElement *h_htTurnOn_num;
  MonitorElement *h_htTurnOn_den;
  MonitorElement *h_photonTurnOn_num;
  MonitorElement *h_photonTurnOn_den;
};

#endif
