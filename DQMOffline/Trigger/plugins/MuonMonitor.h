#ifndef DQMOffline_Trigger_MuonMonitor_h
#define DQMOffline_Trigger_MuonMonitor_h

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/interface/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class MuonMonitor : public DQMEDAnalyzer, public TriggerDQMBase {

 public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  MuonMonitor(const edm::ParameterSet&);
  ~MuonMonitor() throw() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

 private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::PFMETCollection> metToken_;
  edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron>> eleToken_;

  static constexpr double MAX_PHI = 3.2;
  static constexpr int N_PHI = 64;
  const MEbinning phi_binning_{N_PHI, -MAX_PHI, MAX_PHI};

  static constexpr double MAX_dxy = 2.5;
  static constexpr int N_dxy = 50;
  const MEbinning dxy_binning_{N_dxy, -MAX_dxy, MAX_dxy};

  static constexpr double MAX_ETA = 2.4;
  static constexpr int N_ETA = 68;
  const MEbinning eta_binning_{N_ETA, -MAX_ETA, MAX_ETA};

  std::vector<double> muon_variable_binning_;
  std::vector<double> muoneta_variable_binning_;
  MEbinning muon_binning_;
  MEbinning ls_binning_;
  std::vector<double> muPt_variable_binning_2D_;
  std::vector<double> elePt_variable_binning_2D_;
  std::vector<double> muEta_variable_binning_2D_;
  std::vector<double> eleEta_variable_binning_2D_;

  ObjME muonME_;
  ObjME muonEtaME_;
  ObjME muonPhiME_;
  ObjME muonME_variableBinning_;
  ObjME muonVsLS_;
  ObjME muonEtaPhiME_;
  ObjME muondxy_;
  ObjME muondz_;
  ObjME muonEtaME_variableBinning_;
  ObjME eleME_variableBinning_;
  ObjME eleEtaME_;
  ObjME eleEta_muEta_;
  ObjME elePt_muPt_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET, true> metSelection_;
  StringCutObjectSelector<reco::Muon, true> muonSelection_;
  StringCutObjectSelector<reco::GsfElectron, true> eleSelection_;

  unsigned int nmuons_;
  unsigned int nelectrons_;
};

#endif
