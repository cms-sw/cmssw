#ifndef DQMOffline_Trigger_ObjMonitor_h
#define DQMOffline_Trigger_ObjMonitor_h

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
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "DQMOffline/Trigger/plugins/METDQM.h"
#include "DQMOffline/Trigger/plugins/JetDQM.h"
#include "DQMOffline/Trigger/plugins/HTDQM.h"
#include "DQMOffline/Trigger/plugins/HMesonGammaDQM.h"

class ObjMonitor : public DQMEDAnalyzer, public TriggerDQMBase {

 public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  ObjMonitor(const edm::ParameterSet&);
  ~ObjMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

 private:
  bool looseJetId(const double& abseta,
                  const double& NHF,
                  const double& NEMF,
                  const double& CHF,
                  const double& CEMF,
                  const unsigned& NumNeutralParticles,
                  const unsigned& CHM);

  bool tightJetId(const double& abseta,
                  const double& NHF,
                  const double& NEMF,
                  const double& CHF,
                  const double& CEMF,
                  const unsigned& NumNeutralParticles,
                  const unsigned& CHM);

  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::PFMETCollection> metToken_;
  edm::EDGetTokenT<reco::PFJetCollection> jetToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection> muoToken_;
  edm::EDGetTokenT<reco::PhotonCollection> phoToken_;
  edm::EDGetTokenT<reco::TrackCollection> trkToken_;

  //objects to plot
  //add your own with corresponding switch
  bool do_met_;
  METDQM metDQM_;
  bool do_jet_;
  JetDQM jetDQM_;
  bool do_ht_;
  HTDQM htDQM_;
  bool do_hmg_;
  HMesonGammaDQM hmgDQM_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET, true> metSelection_;
  StringCutObjectSelector<reco::PFJet, true> jetSelection_;
  std::string jetId_;
  StringCutObjectSelector<reco::PFJet, true> htjetSelection_;
  StringCutObjectSelector<reco::GsfElectron, true> eleSelection_;
  StringCutObjectSelector<reco::Muon, true> muoSelection_;
  StringCutObjectSelector<reco::Photon, true> phoSelection_;
  StringCutObjectSelector<reco::Track, true> trkSelection_;

  unsigned njets_;
  unsigned nelectrons_;
  unsigned nmuons_;
  unsigned nphotons_;
  unsigned nmesons_;
};

#endif
