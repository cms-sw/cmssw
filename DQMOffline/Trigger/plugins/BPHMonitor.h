#ifndef BPHMONITOR_H
#define BPHMONITOR_H

#include <string>
#include <vector>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/interface/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/TriggerUtils/interface/PrescaleWeightProvider.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"

class BPHMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  BPHMonitor(const edm::ParameterSet&);
  ~BPHMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void case11_selection(const float& dimuonCL,
                        const float& jpsi_cos,
                        const GlobalPoint& displacementFromBeamspotJpsi,
                        const GlobalError& jerr,
                        const edm::Handle<reco::TrackCollection>& trHandle,
                        const std::string& hltpath,
                        const edm::Handle<trigger::TriggerEvent>& handleTriggerEvent,
                        const reco::Muon& m,
                        const reco::Muon& m1,
                        const edm::ESHandle<MagneticField>& bFieldHandle,
                        const reco::BeamSpot& vertexBeamSpot,
                        MonitorElement* phi1,
                        MonitorElement* eta1,
                        MonitorElement* pT1,
                        MonitorElement* phi2,
                        MonitorElement* eta2,
                        MonitorElement* pT2);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

  template <typename T>
  bool matchToTrigger(const std::string& theTriggerName, T t);

  double Prescale(const std::string num,
                  const std::string den,
                  edm::Event const& iEvent,
                  edm::EventSetup const& iSetup,
                  HLTPrescaleProvider* hltPrescale_);

private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::InputTag muoInputTag_;
  edm::InputTag bsInputTag_;
  edm::InputTag trInputTag_;
  edm::InputTag phInputTag_;
  edm::InputTag vtxInputTag_;

  edm::EDGetTokenT<reco::MuonCollection> muoToken_;
  edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  edm::EDGetTokenT<reco::TrackCollection> trToken_;
  edm::EDGetTokenT<reco::PhotonCollection> phToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;

  std::vector<double> pt_variable_binning_;
  std::vector<double> dMu_pt_variable_binning_;
  std::vector<double> prob_variable_binning_;
  MEbinning phi_binning_;
  MEbinning eta_binning_;
  MEbinning d0_binning_;
  MEbinning z0_binning_;
  MEbinning dR_binning_;
  MEbinning mass_binning_;
  MEbinning Bmass_binning_;
  MEbinning dca_binning_;
  MEbinning ds_binning_;
  MEbinning cos_binning_;

  ObjME muPhi_;
  ObjME muEta_;
  ObjME muPt_;
  ObjME mud0_;
  ObjME muz0_;

  ObjME mu1Phi_;
  ObjME mu1Eta_;
  ObjME mu1Pt_;
  ObjME mu1d0_;
  ObjME mu1z0_;
  ObjME mu2Phi_;
  ObjME mu2Eta_;
  ObjME mu2Pt_;
  ObjME mu2d0_;
  ObjME mu2z0_;
  ObjME mu3Phi_;
  ObjME mu3Eta_;
  ObjME mu3Pt_;
  ObjME mu3d0_;
  ObjME mu3z0_;

  ObjME phPhi_;
  ObjME phEta_;
  ObjME phPt_;
  ObjME DiMuPhi_;
  ObjME DiMuEta_;
  ObjME DiMuPt_;
  ObjME DiMuPVcos_;
  ObjME DiMuProb_;
  ObjME DiMuDS_;
  ObjME DiMuDCA_;
  ObjME DiMuMass_;
  ObjME BMass_;
  ObjME DiMudR_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  HLTPrescaleProvider* hltPrescale_;

  StringCutObjectSelector<reco::Muon, true> muoSelection_;
  StringCutObjectSelector<reco::Muon, true> muoSelection_ref;
  StringCutObjectSelector<reco::Muon, true> muoSelection_tag;
  StringCutObjectSelector<reco::Muon, true> muoSelection_probe;

  int nmuons_;
  bool tnp_;
  int L3_;
  int ptCut_;
  int displaced_;
  int trOrMu_;
  int Jpsi_;
  int Upsilon_;
  int enum_;
  int seagull_;
  double maxmass_;
  double minmass_;
  double maxmassJpsi;
  double minmassJpsi;
  double maxmassUpsilon;
  double minmassUpsilon;
  double maxmassTkTk;
  double minmassTkTk;
  double maxmassJpsiTk;
  double minmassJpsiTk;
  double kaon_mass;
  double mu_mass;
  double min_dR;
  double max_dR;

  double minprob;
  double mincos;
  double minDS;
  edm::EDGetTokenT<edm::TriggerResults> hltTrigResTag_;
  edm::InputTag hltInputTag_1;
  edm::EDGetTokenT<trigger::TriggerEvent> hltInputTag_;
  std::vector<std::string> hltpaths_num;
  std::vector<std::string> hltpaths_den;
  StringCutObjectSelector<reco::Track, true> trSelection_;
  StringCutObjectSelector<reco::Track, true> trSelection_ref;
  StringCutObjectSelector<reco::Candidate::LorentzVector, true> DMSelection_ref;

  edm::Handle<trigger::TriggerEvent> handleTriggerEvent;

  HLTConfigProvider hltConfig_;
  edm::Handle<edm::TriggerResults> HLTR;
  std::string getTriggerName(std::string partialName);

  std::vector<bool> warningPrinted4token_;
};

#endif  // METMONITOR_H
