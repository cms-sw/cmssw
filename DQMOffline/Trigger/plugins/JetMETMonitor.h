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

struct JetMETME {
  MonitorElement* numerator;
  MonitorElement* denominator;
};

struct JetEnFracME {
  MonitorElement* Hist;
};

struct JetResponseME {

  MonitorElement* hltpt;
  MonitorElement* hlteta;
  MonitorElement* hltphi;
  MonitorElement* hltenergy;

  MonitorElement* offpt;
  MonitorElement* offeta;
  MonitorElement* offphi;
  MonitorElement* offenergy;
  
};

//
// class declaration
//

class JetMETMonitor : public DQMEDAnalyzer 
{
public:
  JetMETMonitor( const edm::ParameterSet& );
  ~JetMETMonitor();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, JetMETME& me, std::string& histname, std::string& histtitle, int& nbins, double& xmin, double& xmax);
  void bookME(DQMStore::IBooker &, JetMETME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX);
  void bookME(DQMStore::IBooker &, JetMETME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, double& ymin, double& ymax);
  void bookME(DQMStore::IBooker &, JetMETME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, int& nbinsY, double& ymin, double& ymax);
  void bookME(DQMStore::IBooker &, JetMETME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX, std::vector<double> binningY);
  void setMETitle(JetMETME& me, std::string titleX, std::string titleY);
  void setMETitle(JetEnFracME& me, std::string titleX, std::string titleY);
  // Book Histo with JetEnFracME
  void bookJEFME(DQMStore::IBooker &, JetEnFracME& me, std::string& histname, std::string& histtitle, int& nbins, double& xmin, double& xmax);
  void bookJEFME(DQMStore::IBooker &, JetEnFracME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax,int& nbinsY, double& ymin, double& ymax );
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

private:
  static MEbinning getHistoPSet    (edm::ParameterSet pset);
  static MEbinning getHistoLSPSet  (edm::ParameterSet pset);

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::PFMETCollection>       metToken_;
  edm::EDGetTokenT<reco::PFJetCollection>       hltpfjetToken_;// HLT pfJet 
  edm::EDGetTokenT<reco::PFJetCollection>       pfjetToken_;// pfjet
  edm::EDGetTokenT<reco::GenJetCollection>      genjetToken_;// genjet
  edm::EDGetTokenT<reco::CaloJetCollection>     hltcalojetToken_;// HLT calojet
  edm::EDGetTokenT<reco::CaloJetCollection>     calojetToken_;// calojet
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection>        muoToken_;

  std::vector<double> met_variable_binning_;
  MEbinning           met_binning_;
  MEbinning           ls_binning_;

  JetMETME jetmetME_;
  JetMETME jetmetME_variableBinning_;
  JetMETME jetmetVsLS_;
  JetMETME jetmetPhiME_;
  JetMETME jetmetEtaME_;
  JetMETME jetmetEtaVsPhi_;

  /// For Barrel
  JetMETME jetmetHB_ME_;
  JetMETME jetmetHB_ME_variableBinning_;
  JetMETME jetmetHB_VsLS_;
  JetMETME jetmetHB_PhiME_;
  JetMETME jetmetHB_EtaME_;

  /// For Endcap
  JetMETME jetmetHE_ME_;
  JetMETME jetmetHE_ME_variableBinning_;
  JetMETME jetmetHE_VsLS_;
  JetMETME jetmetHE_PhiME_;
  JetMETME jetmetHE_EtaME_;

  /// For Endcap_plus ///
  JetMETME jetmetHE_p_ME_;
  JetMETME jetmetHE_p_ME_variableBinning_;
  JetMETME jetmetHE_p_VsLS_;
  JetMETME jetmetHE_p_PhiME_;
  JetMETME jetmetHE_p_EtaME_;

  /// For Endcap_minus ///
  JetMETME jetmetHE_m_ME_;
  JetMETME jetmetHE_m_ME_variableBinning_;
  JetMETME jetmetHE_m_VsLS_;
  JetMETME jetmetHE_m_PhiME_;
  JetMETME jetmetHE_m_EtaME_;

  /// For Forward ///
  JetMETME jetmetHF_ME_;
  JetMETME jetmetHF_ME_variableBinning_;
  JetMETME jetmetHF_VsLS_;
  JetMETME jetmetHF_PhiME_;
  JetMETME jetmetHF_EtaME_;

  /// For HEP17///
  JetMETME jetmetHEP17_ME_;
  JetMETME jetmetHEP17_ME_variableBinning_;
  JetMETME jetmetHEP17_VsLS_;
  JetMETME jetmetHEP17_PhiME_;
  JetMETME jetmetHEP17_EtaME_;

  /// For HEM17///
  JetMETME jetmetHEM17_ME_;
  JetMETME jetmetHEM17_ME_variableBinning_;
  JetMETME jetmetHEM17_VsLS_;
  JetMETME jetmetHEM17_PhiME_;
  JetMETME jetmetHEM17_EtaME_;

  /// For HEP18///
  JetMETME jetmetHEP18_ME_;
  JetMETME jetmetHEP18_ME_variableBinning_;
  JetMETME jetmetHEP18_VsLS_;
  JetMETME jetmetHEP18_PhiME_;
  JetMETME jetmetHEP18_EtaME_;

  JetMETME hltjetmetME_;
  JetMETME hltjetmetME_variableBinning_;
  JetMETME hltjetmetVsLS_;
  JetMETME hltjetmetPhiME_;
  JetMETME hltjetmetEtaME_;

  GenericTriggerEventFlag* num_genTriggerEventFlag_;
  GenericTriggerEventFlag* den_genTriggerEventFlag_;

  JetEnFracME hltjetJetFracME_NHEF; 
  JetEnFracME hltjetJetFracME_NEEF; 
  JetEnFracME hltjetJetFracME_CHEF; 
  JetEnFracME hltjetJetFracME_MuEF; 
  JetEnFracME hltjetJetFracME_CEEF;
  // 2-D plots 
  JetEnFracME hltjetJetFracME_NHEFVshltjetPt; 
  JetEnFracME hltjetJetFracME_NEEFVshltjetPt; 
  JetEnFracME hltjetJetFracME_CHEFVshltjetPt; 
  JetEnFracME hltjetJetFracME_MuEFVshltjetPt; 
  JetEnFracME hltjetJetFracME_CEEFVshltjetPt; 

  int njets_;
  int nelectrons_;
  int nmuons_;
  double ptcut_;
  bool isPFJetTrig;
  bool isCaloJetTrig;
  bool isMetTrig;
  bool isJetFrac;

};

#endif // JETMETMONITOR_H
