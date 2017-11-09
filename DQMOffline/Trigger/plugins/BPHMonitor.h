#ifndef BPHMONITOR_H
#define BPHMONITOR_H

#include <string>
#include <vector>
#include <map>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Math/interface/deltaR.h"
//DataFormats
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h" 
#include "DataFormats/VertexReco/interface/Vertex.h" 
#include "DataFormats/VertexReco/interface/VertexFwd.h" 
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "CommonTools/TriggerUtils/interface/PrescaleWeightProvider.h"

class GenericTriggerEventFlag;

struct MEbinning {
  MEbinning(int n, double min, double max) : nbins(n), xmin(min), xmax(max) {}
  MEbinning(std::vector<double> e) : edges(std::move(e)) {}
  int nbins;
  double xmin;
  double xmax;
  std::vector<double> edges;
};

struct  METME{
  MonitorElement* numerator;
  MonitorElement* denominator;
};

//
// class declaration
//

class BPHMonitor : public DQMEDAnalyzer 
{
public:
  BPHMonitor( const edm::ParameterSet& );
  ~BPHMonitor() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

  void case11_selection(const float & dimuonCL, const float & jpsi_cos, const GlobalPoint & displacementFromBeamspotJpsi, const GlobalError & jerr, const edm::Handle<reco::TrackCollection> & trHandle, const std::string & hltpath, const edm::Handle<trigger::TriggerEvent> & handleTriggerEvent, const reco::Muon& m, const reco::Muon& m1, const edm::ESHandle<MagneticField> & bFieldHandle, const reco::BeamSpot & vertexBeamSpot, MonitorElement* phi1, MonitorElement* eta1, MonitorElement* pT1, MonitorElement* phi2, MonitorElement* eta2, MonitorElement* pT2);


protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, METME& me, std::string& histname, std::string& histtitle, int& nbins, double& xmin, double& xmax);
  void bookME(DQMStore::IBooker &, METME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX);
  void bookME(DQMStore::IBooker &, METME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, double& ymin, double& ymax);
  void bookME(DQMStore::IBooker &, METME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, int& nbinsY, double& ymin, double& ymax);
  void bookME(DQMStore::IBooker &, METME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX, std::vector<double> binningY);
  void bookME(DQMStore::IBooker &, METME& me, std::string &histname, std::string &histtitle, /*const*/ MEbinning& binning);

  void setMETitle(METME& me, std::string titleX, std::string titleY);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  template <typename T>
  bool matchToTrigger(const std::string &theTriggerName ,T t, edm::Handle<trigger::TriggerEvent> handleTriggerEvent);
  //bool matchToTrigger(std::string theTriggerName,T t, edm::Handle<trigger::TriggerEventWithRefs> handleTriggerEvent);


private:
  static MEbinning getHistoPSet    (edm::ParameterSet pset);
  // static MEbinning getHistoLSPSet  (edm::ParameterSet pset);

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::MuonCollection>        muoToken_;
  edm::EDGetTokenT<reco::BeamSpot>        bsToken_;
  edm::EDGetTokenT<reco::TrackCollection>       trToken_;
  edm::EDGetTokenT<reco::PhotonCollection>       phToken_;

  MEbinning           phi_binning_;
  MEbinning           pt_binning_;
  MEbinning           eta_binning_;
  MEbinning           d0_binning_;
  MEbinning           z0_binning_;
  MEbinning           dR_binning_;
  MEbinning           mass_binning_;
  MEbinning           dca_binning_;
  MEbinning           ds_binning_;
  MEbinning           cos_binning_;
  MEbinning           prob_binning_;

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


///
  METME  phPhi_   ;
  METME  phEta_   ;
  METME  phPt_   ;
  METME  DiMuPhi_   ;
  METME  DiMuEta_   ;
  METME  DiMuPt_   ;
  METME  DiMuPVcos_   ;
  METME  DiMuProb_   ;
  METME  DiMuDS_   ;
  METME  DiMuDCA_   ;
  METME  DiMuMass_   ;
  METME  DiMudR_   ;


//

  GenericTriggerEventFlag* num_genTriggerEventFlag_;
  GenericTriggerEventFlag* den_genTriggerEventFlag_;
  PrescaleWeightProvider * prescaleWeightProvider_;
  StringCutObjectSelector<reco::Muon,true>        muoSelection_;
  StringCutObjectSelector<reco::Muon,true>        muoSelection_ref;
  StringCutObjectSelector<reco::Muon,true>        muoSelection_tag;
  StringCutObjectSelector<reco::Muon,true>        muoSelection_probe;
  int nmuons_;
  bool tnp_;
  int L3_;
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

  double minprob;
  double mincos;
  double minDS;
  edm::EDGetTokenT<edm::TriggerResults>  hltTrigResTag_;
  edm::EDGetTokenT<trigger::TriggerEvent>  hltInputTag_;
  std::vector<std::string> hltpaths_num;
  std::vector<std::string> hltpaths_den;
  StringCutObjectSelector<reco::Track,true>        trSelection_;
  StringCutObjectSelector<reco::Track,true>        trSelection_ref;
  StringCutObjectSelector<reco::Candidate::LorentzVector,true>        DMSelection_ref;

};

#endif // METMONITOR_H
