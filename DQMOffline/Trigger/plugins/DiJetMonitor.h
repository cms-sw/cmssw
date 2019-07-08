#ifndef DIJETMETMONITOR_H
#define DIJETMETMONITOR_H

#include <string>
#include <vector>
#include <map>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

//DataFormats
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

class GenericTriggerEventFlag;

//
// class declaration
//

class DiJetMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  DiJetMonitor(const edm::ParameterSet &);
  ~DiJetMonitor() throw() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup) override;
  bool dijet_selection(double eta_1,
                       double phi_1,
                       double eta_2,
                       double phi_2,
                       double pt_1,
                       double pt_2,
                       int &tag_id,
                       int &probe_id,
                       int Event);

private:
  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::PFMETCollection> metToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection> muoToken_;
  edm::EDGetTokenT<reco::PFJetCollection> dijetSrc_;  // test for Jet

  MEbinning dijetpt_binning_;
  MEbinning dijetptThr_binning_;

  ObjME jetpt1ME_;
  ObjME jetpt2ME_;
  ObjME jetptAvgaME_;
  ObjME jetptAvgaThrME_;
  ObjME jetptAvgbME_;
  ObjME jetptTagME_;
  ObjME jetptPrbME_;
  ObjME jetptAsyME_;
  ObjME jetetaPrbME_;
  ObjME jetetaTagME_;
  ObjME jetphiPrbME_;
  ObjME jetAsyEtaME_;
  ObjME jetEtaPhiME_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  int nmuons_;
  double ptcut_;

  // Define Phi Bin //
  const double DiJet_MAX_PHI = 3.2;
  //unsigned int DiJet_N_PHI = 64;
  unsigned int DiJet_N_PHI = 32;
  MEbinning dijet_phi_binning{DiJet_N_PHI, -DiJet_MAX_PHI, DiJet_MAX_PHI};
  // Define Eta Bin //
  const double DiJet_MAX_ETA = 5;
  //unsigned int DiJet_N_ETA = 50;
  unsigned int DiJet_N_ETA = 20;
  MEbinning dijet_eta_binning{DiJet_N_ETA, -DiJet_MAX_ETA, DiJet_MAX_ETA};

  const double MAX_asy = 1;
  const double MIN_asy = -1;
  //unsigned int N_asy = 100;
  unsigned int N_asy = 50;
  MEbinning asy_binning{N_asy, MIN_asy, MAX_asy};
};

#endif  // DIJETMETMONITOR_H
