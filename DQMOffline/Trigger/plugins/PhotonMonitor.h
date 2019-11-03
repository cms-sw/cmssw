#ifndef DQMOFFLINE_TRIGGER_PHOTON_H
#define DQMOFFLINE_TRIGGER_PHOTON_H

#include <string>
#include <vector>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
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
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
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

class PhotonMonitor : public DQMEDAnalyzer, public TriggerDQMBase {

 public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  PhotonMonitor(const edm::ParameterSet&);
  ~PhotonMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

 private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::PFMETCollection> metToken_;
  edm::EDGetTokenT<reco::PFJetCollection> jetToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::PhotonCollection> photonToken_;

  std::vector<double> photon_variable_binning_;
  std::vector<double> diphoton_mass_binning_;

  MEbinning photon_binning_;
  MEbinning ls_binning_;

  ObjME subphotonEtaME_;
  ObjME subphotonME_;
  ObjME subphotonPhiME_;
  ObjME subphotonME_variableBinning_;
  ObjME subphotonEtaPhiME_;
  ObjME subphotonr9ME_;
  ObjME subphotonHoverEME_;
  ObjME diphotonMassME_;

  ObjME photonEtaME_;
  ObjME photonME_;
  ObjME photonPhiME_;
  ObjME photonME_variableBinning_;
  ObjME photonVsLS_;
  ObjME photonEtaPhiME_;
  ObjME photonr9ME_;
  ObjME photonHoverEME_;

  double MAX_PHI1 = 3.2;
  unsigned int N_PHI1 = 64;
  const MEbinning phi_binning_1{N_PHI1, -MAX_PHI1, MAX_PHI1};

  double MAX_ETA = 1.4442;
  unsigned int N_ETA = 34;
  const MEbinning eta_binning_{N_ETA, -MAX_ETA, MAX_ETA};

  double MAX_r9 = 1;
  double MIN_r9 = 0;
  unsigned int N_r9 = 50;
  const MEbinning r9_binning_{N_r9, MIN_r9, MAX_r9};

  double MAX_hoe = 0.02;
  double MIN_hoe = 0;
  const MEbinning hoe_binning_{N_r9, MIN_hoe, MAX_hoe};

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET, true> metSelection_;
  StringCutObjectSelector<reco::PFJet, true> jetSelection_;
  StringCutObjectSelector<reco::GsfElectron, true> eleSelection_;
  StringCutObjectSelector<reco::Photon, true> photonSelection_;
  unsigned int njets_;
  unsigned int nphotons_;
  unsigned int nelectrons_;
};

#endif
