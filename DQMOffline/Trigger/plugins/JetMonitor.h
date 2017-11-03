#ifndef JETMETMONITOR_H
#define JETMETMONITOR_H

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

class JetMonitor : public DQMEDAnalyzer 
{
public:
  struct MEbinning {
    unsigned int nbins;
    double xmin;
    double xmax;
  };

  struct JetME {
    MonitorElement* numerator = nullptr;
    MonitorElement* denominator= nullptr;
  };

  JetMonitor( const edm::ParameterSet& );
  ~JetMonitor() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, JetME& me, std::string& histname, std::string& histtitle, unsigned int nbins, double xmin, double xmax);
  void bookME(DQMStore::IBooker &, JetME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX);
  void bookME(DQMStore::IBooker &, JetME& me, std::string& histname, std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, JetME& me, std::string& histname, std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, JetME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX, std::vector<double> binningY);
  void setMETitle(JetME& me, const std::string& titleX, const std::string& titleY);
  void bookME(DQMStore::IBooker &, MonitorElement* me, std::string& histname, std::string& histtitle, int nbins, double xmin, double xmax);
  void bookME(DQMStore::IBooker &, MonitorElement* me, std::string& histname, std::string& histtitle, int nbinsX, double xmin, double xmax,int nbinsY, double ymin, double ymax );

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  bool isBarrel(double eta);
  bool isEndCapP(double eta);
  bool isEndCapM(double eta);
  bool isForward(double eta);
  bool isHEP17(double eta, double phi);
  bool isHEM17(double eta, double phi);
  bool isHEP18(double eta, double phi); // -0.87< Phi < -1.22 
  //void FillME(std::vector<MonitorElement*> v_me,std::vector<double>v_pt, std::vector<double> v_phi); //Fill Histograms 
  //void AutoNullPtr(JetME* a_me,const int len_); //Fill Histograms 
  void bookMESub(DQMStore::IBooker &,JetME* a_me,const int len_,const std::string& h_Name ,const std::string& h_Title, const std::string& h_subOptName, std::string h_subOptTitle, bool doPhi = true, bool doEta = true, bool doEtaPhi = true, bool doVsLS = true); //Fill Histograms 
  void FillME(JetME* a_me,double pt_, double phi_, double eta_, int ls_, const std::string& denu, bool doPhi = true, bool doEta = true, bool doEtaPhi = true, bool doVsLS = true); //Fill Histograms 

private:
  static MEbinning getHistoPSet    (const edm::ParameterSet& pset);
  static MEbinning getHistoLSPSet  (const edm::ParameterSet& pset);

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::PFMETCollection>       metToken_;
  edm::EDGetTokenT<reco::PFJetCollection>       pfjetToken_;// pfjet
  edm::EDGetTokenT<reco::CaloJetCollection>     calojetToken_;// calojet
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection>        muoToken_;
  //edm::InputTag        jetSrc_; // test for Jet
  edm::EDGetTokenT<edm::View<reco::Jet>  >  jetSrc_; // test for Jet

  std::vector<double> jetpT_variable_binning_;
  MEbinning           jetpT_binning;
  MEbinning           jetptThr_binning_;
  MEbinning           ls_binning_;


  JetME a_ME[7];
  JetME a_ME_HB[7];
  JetME a_ME_HE[7];
  JetME a_ME_HF[7];
  JetME a_ME_HE_p[7];
  JetME a_ME_HE_m[7];
  JetME a_ME_HEM17[7];
  JetME a_ME_HEP17[7];
  JetME a_ME_HEP18[7];

  JetME jetHEP17_AbsEtaVsPhi_;
  JetME jetHEM17_AbsEtaVsPhi_;
  // For Ratio plot 
  JetME jetHEP17_AbsEta_;
  JetME jetHEM17_AbsEta_;


  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  int nmuons_;
  double ptcut_;
  bool isPFJetTrig;
  bool isCaloJetTrig;

  std::vector<double> v_jetpt;
  std::vector<double> v_jeteta;
  std::vector<double> v_jetphi;

  // Define Phi Bin //
  double Jet_MAX_PHI = 3.2;
  //  unsigned int Jet_N_PHI = 64;
  unsigned int Jet_N_PHI = 32;
  MEbinning jet_phi_binning_{
    Jet_N_PHI, -Jet_MAX_PHI, Jet_MAX_PHI
  };
  // Define Eta Bin //
  double Jet_MAX_ETA = 5;
  //  unsigned int Jet_N_ETA = 50;
  unsigned int Jet_N_ETA = 20; // (mia) not optimal, we should make use of variable binning which reflects the detector !
  MEbinning jet_eta_binning_{
    Jet_N_ETA, -Jet_MAX_ETA, Jet_MAX_ETA
  };

  // Define HEP17 for PHI 
  double Jet_MAX_PHI_HEP17 = -0.52;
  double Jet_MIN_PHI_HEP17 = -0.87;
  unsigned int N_PHI_HEP17 = 7;
  MEbinning phi_binning_hep17_{
    N_PHI_HEP17, Jet_MIN_PHI_HEP17, Jet_MAX_PHI_HEP17
  };

  // Define HEP18 for PHI 
  double Jet_MAX_PHI_HEP18 = -0.17;
  double Jet_MIN_PHI_HEP18 = -0.52;
  unsigned int N_PHI_HEP18 = 7;
  MEbinning phi_binning_hep18_{
    N_PHI_HEP18, Jet_MIN_PHI_HEP18, Jet_MAX_PHI_HEP18
  };
  // Define HEP17 for ETA 
  double Jet_MAX_ETA_HEP17 = 3.0;
  double Jet_MIN_ETA_HEP17 = 1.3;
  unsigned int N_ETA_HEP17 = 9;
  MEbinning eta_binning_hep17_{
    N_ETA_HEP17, Jet_MIN_ETA_HEP17, Jet_MAX_ETA_HEP17
  };

  MEbinning eta_binning_hem17_{
    N_ETA_HEP17, -Jet_MAX_ETA_HEP17, -Jet_MIN_ETA_HEP17
  };

};

#endif // JETMETMONITOR_H
