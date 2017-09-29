#ifndef DQMOffline_Trigger_MuonMonitor_h
#define DQMOffline_Trigger_MuonMonitor_h

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

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

//DataFormats
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
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class GenericTriggerEventFlag;

struct MEbinning {
  unsigned int nbins;
  double xmin;
  double xmax;
};

struct MuonME {
  MonitorElement* numerator;
  MonitorElement* denominator;
};
//
// class declaration
//

class MuonMonitor : public DQMEDAnalyzer 
{
public:
  MuonMonitor( const edm::ParameterSet& );
  ~MuonMonitor() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, MuonME& me, const std::string& histname, const std::string& histtitle, int nbins, double xmin, double xmax);
  void bookME(DQMStore::IBooker &, MuonME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX);
  void bookME(DQMStore::IBooker &, MuonME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, MuonME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, MuonME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY);
  void setTitle(MuonME& me,  const std::string& titleX,  const std::string& titleY);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  static MEbinning getHistoPSet    (edm::ParameterSet const& pset);
  static MEbinning getHistoLSPSet  (edm::ParameterSet const& pset);

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::PFMETCollection>        metToken_;
  edm::EDGetTokenT<reco::MuonCollection>         muonToken_;
  edm::EDGetTokenT<reco::VertexCollection>       vtxToken_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron>> eleToken_;

  static constexpr double MAX_PHI = 3.2;
  static constexpr int    N_PHI = 64;
  const MEbinning phi_binning_{
    N_PHI, -MAX_PHI, MAX_PHI
  };

  static constexpr double MAX_dxy = 2.5;
  static constexpr int    N_dxy = 50;
  const MEbinning dxy_binning_{
    N_dxy, -MAX_dxy, MAX_dxy
  };

  static constexpr double MAX_ETA = 2.4;
  static constexpr int    N_ETA = 68;
  const MEbinning eta_binning_{
    N_ETA, -MAX_ETA, MAX_ETA
  };

  std::vector<double> muon_variable_binning_;
  std::vector<double> muoneta_variable_binning_;
  MEbinning           muon_binning_;
  MEbinning           ls_binning_;
  std::vector<double> muPt_variable_binning_2D_;
  std::vector<double> elePt_variable_binning_2D_;
  std::vector<double> muEta_variable_binning_2D_;
  std::vector<double> eleEta_variable_binning_2D_;
  
  MuonME muonME_;
  MuonME muonEtaME_;
  MuonME muonPhiME_;
  MuonME muonME_variableBinning_;
  MuonME muonVsLS_;
  MuonME muonEtaPhiME_;   
  MuonME muondxy_;   
  MuonME muondz_;   
  MuonME muonEtaME_variableBinning_;
  MuonME eleME_variableBinning_;
  MuonME eleEtaME_;
  MuonME eleEta_muEta_;
  MuonME elePt_muPt_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET,true>         metSelection_;
  StringCutObjectSelector<reco::Muon,true> muonSelection_;
  StringCutObjectSelector<reco::GsfElectron,true> eleSelection_;
  unsigned int njets_;
  unsigned int nmuons_;
  unsigned int nelectrons_;
};

#endif // DQMOffline_Trigger_MuonMonitor_h
