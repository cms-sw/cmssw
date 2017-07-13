#ifndef DQMOffline_Trigger_BPHMonitor_h
#define DQMOffline_Trigger_BPHMonitor_h

#include <map>
#include <string>
#include <vector>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class GenericTriggerEventFlag;

struct MEbinning
{
  int nbins;
  double xmin;
  double xmax;
};

struct METME
{
  MonitorElement* numerator;
  MonitorElement* denominator;
};

//
// class declaration
//

class BPHMonitor : public DQMEDAnalyzer
{
public:
  BPHMonitor(const edm::ParameterSet&);
  ~BPHMonitor() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription& pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription& pset);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void bookME(DQMStore::IBooker&, METME& me, std::string& histname, std::string& histtitle, int& nbins, double& xmin, double& xmax);
  void bookME(DQMStore::IBooker&, METME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX);
  void bookME(DQMStore::IBooker&, METME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, double& ymin, double& ymax);
  void bookME(DQMStore::IBooker&, METME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, int& nbinsY, double& ymin, double& ymax);
  void bookME(DQMStore::IBooker&, METME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX, std::vector<double> binningY);
  void setMETitle(METME& me, const std::string& titleX, const std::string& titleY);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  template <typename T>
  bool matchToTrigger(const std::string& theTriggerName, T t, const edm::Handle<trigger::TriggerEvent>& handleTriggerEvent);

private:
  static MEbinning getHistoPSet(const edm::ParameterSet& pset);
  static MEbinning getHistoLSPSet(const edm::ParameterSet& pset);

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::MuonCollection> muoToken_;
  edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  edm::EDGetTokenT<reco::TrackCollection> trToken_;
  edm::EDGetTokenT<reco::PhotonCollection> phToken_;

  MEbinning phi_binning_;
  MEbinning pt_binning_;
  MEbinning eta_binning_;
  MEbinning d0_binning_;
  MEbinning z0_binning_;
  MEbinning dR_binning_;
  MEbinning mass_binning_;
  MEbinning dca_binning_;
  MEbinning ds_binning_;
  MEbinning cos_binning_;
  MEbinning prob_binning_;

  METME muPhi_;
  METME muEta_;
  METME muPt_;
  METME mud0_;
  METME muz0_;

  METME mu1Phi_;
  METME mu1Eta_;
  METME mu1Pt_;
  METME mu1d0_;
  METME mu1z0_;
  METME mu2Phi_;
  METME mu2Eta_;
  METME mu2Pt_;
  METME mu2d0_;
  METME mu2z0_;
  METME mu3Phi_;
  METME mu3Eta_;
  METME mu3Pt_;
  METME mu3d0_;
  METME mu3z0_;

  METME phPhi_;
  METME phEta_;
  METME phPt_;
  METME diMuPhi_;
  METME diMuEta_;
  METME diMuPt_;
  METME diMuPVcos_;
  METME diMuProb_;
  METME diMuDS_;
  METME diMuDCA_;
  METME diMuMass_;
  METME diMudR_;

  GenericTriggerEventFlag* num_genTriggerEventFlag_;
  GenericTriggerEventFlag* den_genTriggerEventFlag_;
  StringCutObjectSelector<reco::Muon, true> muoSelection_;
  StringCutObjectSelector<reco::Muon, true> muoSelection_ref_;
  StringCutObjectSelector<reco::Muon, true> muoSelection_tag_;
  StringCutObjectSelector<reco::Muon, true> muoSelection_probe_;
  int nmuons_;
  int tnp_;
  int l3_;
  int trOrMu_;
  int jpsi_;
  int upsilon_;
  int nofset_;
  int seagull_;
  double maxmass_;
  double minmass_;
  double maxmassJpsi_;
  double minmassJpsi_;
  double maxmassUpsilon_;
  double minmassUpsilon_;
  double maxmassJpsiTk_;
  double minmassJpsiTk_;
  double minprob_;
  double mincos_;
  double minDS_;
  edm::EDGetTokenT<trigger::TriggerEvent> hltInputTag_;
  std::vector<std::string> hltpaths_num_;
  std::vector<std::string> hltpaths_den_;
  StringCutObjectSelector<reco::Track, true> trSelection_;
  StringCutObjectSelector<reco::Track, true> trSelection_ref_;
  StringCutObjectSelector<reco::Candidate::LorentzVector, true> dmSelection_ref_;
};

#endif  // DQMOffline_Trigger_BPHMonitor_h
