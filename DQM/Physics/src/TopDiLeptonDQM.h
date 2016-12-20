#ifndef TopDiLeptonDQM_H
#define TopDiLeptonDQM_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class TH1F;
class TH2F;

class TopDiLeptonDQM : public DQMEDAnalyzer {
 public:
  explicit TopDiLeptonDQM(const edm::ParameterSet&);
  ~TopDiLeptonDQM();

 protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&,
                      edm::EventSetup const&) override;
 private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  std::string moduleName_;
  std::string outputFile_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
  std::vector<std::string> hltPaths_;
  std::vector<std::string> hltPaths_sig_;
  std::vector<std::string> hltPaths_trig_;

  std::ofstream outfile;

  int N_sig[100];
  int N_trig[100];
  float Eff[100];

  int N_mumu;
  int N_muel;
  int N_elel;

  edm::EDGetTokenT<reco::VertexCollection> vertex_;
  double vertex_X_cut_;
  double vertex_Y_cut_;
  double vertex_Z_cut_;

  edm::EDGetTokenT<reco::MuonCollection> muons_;
  double muon_pT_cut_;
  double muon_eta_cut_;
  double muon_iso_cut_;

  edm::EDGetTokenT<reco::GsfElectronCollection> elecs_;
  double elec_pT_cut_;
  double elec_eta_cut_;
  double elec_iso_cut_;
  double elec_emf_cut_;

  double MassWindow_up_;
  double MassWindow_down_;

  MonitorElement* Events_;
  MonitorElement* Trigs_;
  MonitorElement* TriggerEff_;
  MonitorElement* Ntracks_;

  MonitorElement* Nmuons_;
  MonitorElement* Nmuons_iso_;
  MonitorElement* Nmuons_charge_;
  MonitorElement* VxVy_muons_;
  MonitorElement* Vz_muons_;
  MonitorElement* pT_muons_;
  MonitorElement* eta_muons_;
  MonitorElement* phi_muons_;

  MonitorElement* Nelecs_;
  MonitorElement* Nelecs_iso_;
  MonitorElement* Nelecs_charge_;
  MonitorElement* HoverE_elecs_;
  MonitorElement* pT_elecs_;
  MonitorElement* eta_elecs_;
  MonitorElement* phi_elecs_;

  MonitorElement* MuIso_emEt03_;
  MonitorElement* MuIso_hadEt03_;
  MonitorElement* MuIso_hoEt03_;
  MonitorElement* MuIso_nJets03_;
  MonitorElement* MuIso_nTracks03_;
  MonitorElement* MuIso_sumPt03_;
  MonitorElement* MuIso_CombRelIso03_;

  MonitorElement* ElecIso_cal_;
  MonitorElement* ElecIso_trk_;
  MonitorElement* ElecIso_CombRelIso_;

  MonitorElement* dimassRC_;
  MonitorElement* dimassWC_;
  MonitorElement* dimassRC_LOGX_;
  MonitorElement* dimassWC_LOGX_;
  MonitorElement* dimassRC_LOG10_;
  MonitorElement* dimassWC_LOG10_;

  MonitorElement* D_eta_muons_;
  MonitorElement* D_phi_muons_;
  MonitorElement* D_eta_elecs_;
  MonitorElement* D_phi_elecs_;
  MonitorElement* D_eta_lepts_;
  MonitorElement* D_phi_lepts_;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
