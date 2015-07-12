#ifndef HotlineDQM_H
#define HotlineDQM_H

//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//Muon
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

// MET
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

// Jets
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

//Photons
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class HotlineDQM: public DQMEDAnalyzer{

  public:
  HotlineDQM(const edm::ParameterSet& ps);
  virtual ~HotlineDQM();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup);

  private:
  //variables from config file
  edm::EDGetTokenT<reco::MuonCollection> theMuonCollection_;
  edm::EDGetTokenT<reco::PFMETCollection> thePfMETCollection_;
  edm::EDGetTokenT<reco::CaloMETCollection> theMETCollection_;
  edm::EDGetTokenT<reco::CaloJetCollection> theCaloJetCollection_;
  edm::EDGetTokenT<reco::PhotonCollection> thePhotonCollection_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
  edm::EDGetTokenT<trigger::TriggerEvent> theTrigSummary_;

  std::string triggerPath_;
  edm::InputTag triggerFilter_;

  bool useMuons, useMet, usePFMet, useHT, usePhotons;

  // Histograms
  MonitorElement* h_MuPt;
  MonitorElement* h_PhotonPt;
  MonitorElement* h_HT;
  MonitorElement* h_MetPt;
  MonitorElement* h_PFMetPt;

  MonitorElement* h_OnlineMuPt;
  MonitorElement* h_OnlinePhotonPt;
  MonitorElement* h_OnlineHT;
  MonitorElement* h_OnlineMetPt;
  MonitorElement* h_OnlinePFMetPt;
};

#endif
