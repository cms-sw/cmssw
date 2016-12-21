#ifndef SUSY_HLT_PhotonCaloHT_H
#define SUSY_HLT_PhotonCaloHT_H

//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// MET
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

// Jets
#include "DataFormats/JetReco/interface/CaloJet.h"

// Photon
#include "DataFormats/EgammaCandidates/interface/Photon.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

namespace reco{
    typedef std::vector<reco::Photon> PhotonCollection;
}

class SUSY_HLT_PhotonCaloHT: public DQMEDAnalyzer{

  public:
  SUSY_HLT_PhotonCaloHT(const edm::ParameterSet& ps);
  virtual ~SUSY_HLT_PhotonCaloHT();

  protected:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) ;
  void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  private:
  //histos booking function
  void bookHistos(DQMStore::IBooker &);

  //variables from config file
  edm::EDGetTokenT<reco::CaloJetCollection> theCaloJetCollection_;
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
  MonitorElement* h_photonPt;
  MonitorElement* h_ht;
  MonitorElement* h_htTurnOn_num;
  MonitorElement* h_htTurnOn_den;
  MonitorElement* h_photonTurnOn_num;
  MonitorElement* h_photonTurnOn_den;

};

#endif
