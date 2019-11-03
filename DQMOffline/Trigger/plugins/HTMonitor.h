#ifndef DQMOffline_Trigger_HTMonitor_h
#define DQMOffline_Trigger_HTMonitor_h

#include <string>
#include <vector>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/interface/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class HTMonitor : public DQMEDAnalyzer, public TriggerDQMBase {

 public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  HTMonitor(const edm::ParameterSet&);
  ~HTMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

 private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::InputTag metInputTag_;
  edm::InputTag jetInputTag_;
  edm::InputTag eleInputTag_;
  edm::InputTag muoInputTag_;
  edm::InputTag vtxInputTag_;

  edm::EDGetTokenT<reco::PFMETCollection> metToken_;
  edm::EDGetTokenT<reco::JetView> jetToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection> muoToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;

  std::vector<double> ht_variable_binning_;
  MEbinning ht_binning_;
  MEbinning ls_binning_;

  ObjME qME_variableBinning_;
  ObjME htVsLS_;
  ObjME deltaphimetj1ME_;
  ObjME deltaphij1j2ME_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET, true> metSelection_;
  StringCutObjectSelector<reco::Jet, true> jetSelection_;
  StringCutObjectSelector<reco::GsfElectron, true> eleSelection_;
  StringCutObjectSelector<reco::Muon, true> muoSelection_;
  StringCutObjectSelector<reco::Jet, true> jetSelection_HT_;
  unsigned njets_;
  unsigned nelectrons_;
  unsigned nmuons_;
  double dEtaCut_;

  static constexpr double MAXedge_PHI = 3.2;
  static constexpr int Nbin_PHI = 64;
  static constexpr MEbinning phi_binning_{Nbin_PHI, -MAXedge_PHI, MAXedge_PHI};

  bool warningWasPrinted_;

  enum quant { HT, MJJ, SOFTDROP };
  quant quantity_;
};

#endif
