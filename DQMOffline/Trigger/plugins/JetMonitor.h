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

struct MEbinning {
  int nbins;
  double xmin;
  double xmax;
};

struct JetME {
  MonitorElement* numerator;
  MonitorElement* denominator;
};

//
// class declaration
//

class JetMonitor : public DQMEDAnalyzer 
{
public:
  JetMonitor( const edm::ParameterSet& );
  ~JetMonitor();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, JetME& me, std::string& histname, std::string& histtitle, int& nbins, double& xmin, double& xmax);
  void bookME(DQMStore::IBooker &, JetME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX);
  void bookME(DQMStore::IBooker &, JetME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, double& ymin, double& ymax);
  void bookME(DQMStore::IBooker &, JetME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, int& nbinsY, double& ymin, double& ymax);
  void bookME(DQMStore::IBooker &, JetME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX, std::vector<double> binningY);
  void setMETitle(JetME& me, std::string titleX, std::string titleY);
  void bookME(DQMStore::IBooker &, MonitorElement* me, std::string& histname, std::string& histtitle, int& nbins, double& xmin, double& xmax);
  void bookME(DQMStore::IBooker &, MonitorElement* me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax,int& nbinsY, double& ymin, double& ymax );

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  bool isBarrel(double eta);
  bool isEndCapP(double eta);
  bool isEndCapM(double eta);
  bool isForward(double eta);
  bool isHEP17(double eta, double phi);
  bool isHEM17(double eta, double phi);
  bool isHEP18(double eta, double phi); // -0.87< Phi < -1.22 
  //void FillME(std::vector<MonitorElement*> v_me,std::vector<double>v_pt, std::vector<double> v_phi); //Fill Histograms 
  void AutoNullPtr(JetME* a_me,const int len_); //Fill Histograms 
  void bookMESub(DQMStore::IBooker &,JetME* a_me,const int len_,std::string h_Name ,std::string h_Title, std::string h_subOptName, std::string h_subOptTitle ); //Fill Histograms 
  void FillME(std::vector<JetME> v_me,double pt, double phi, double eta, int ls_, std::string denu); //Fill Histograms 
  void FillME(JetME* a_me,double pt_, double phi_, double eta_, int ls_, std::string denu); //Fill Histograms 

private:
  static MEbinning getHistoPSet    (edm::ParameterSet pset);
  static MEbinning getHistoLSPSet  (edm::ParameterSet pset);

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::PFMETCollection>       metToken_;
  edm::EDGetTokenT<reco::PFJetCollection>       pfjetToken_;// pfjet
  edm::EDGetTokenT<reco::CaloJetCollection>     calojetToken_;// calojet
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection>        muoToken_;

  std::vector<double> met_variable_binning_;
  MEbinning           met_binning_;
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

//  JetME jetHEP18_EtaVsPhi_;

  GenericTriggerEventFlag* num_genTriggerEventFlag_;
  GenericTriggerEventFlag* den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET,true>         metSelection_;
  StringCutObjectSelector<reco::PFJet,true>       jetSelection_;
  StringCutObjectSelector<reco::CaloJet,true>     calojetSelection_;
  StringCutObjectSelector<reco::GsfElectron,true> eleSelection_;
  StringCutObjectSelector<reco::Muon,true>        muoSelection_;

  int njets_;
  int nelectrons_;
  int nmuons_;
  double ptcut_;
  bool isPFJetTrig;
  bool isCaloJetTrig;
  bool isMetTrig;
  bool isJetFrac;
  std::vector<double> v_jetpt;
  std::vector<double> v_jeteta;
  std::vector<double> v_jetphi;
};

#endif // JETMETMONITOR_H
