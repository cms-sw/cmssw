#ifndef TOPMONITOR_H
#define TOPMONITOR_H

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

//root
#include "TLorentzVector.h"


class GenericTriggerEventFlag;

struct MEbinning {
  int nbins;
  double xmin;
  double xmax;
};

struct METME {
  MonitorElement* numerator;
  MonitorElement* denominator;
};
//
// class declaration
//

class TopMonitor : public DQMEDAnalyzer 
{
public:
  TopMonitor( const edm::ParameterSet& );
  ~TopMonitor();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, METME& me, const std::string& histname, const std::string& histtitle, int nbins, double xmin, double xmax);
  void bookME(DQMStore::IBooker &, METME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX);
  void bookME(DQMStore::IBooker &, METME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, METME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, METME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY);
  void setMETitle(METME& me, std::string titleX, std::string titleY);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  static MEbinning getHistoPSet    (edm::ParameterSet pset);
  static MEbinning getHistoLSPSet  (edm::ParameterSet pset);

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::PFMETCollection>       metToken_;
  edm::EDGetTokenT<reco::PFJetCollection>       jetToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection>        muoToken_;

  MEbinning           met_binning_;
  MEbinning           ls_binning_;
  MEbinning           phi_binning_;
  MEbinning           pt_binning_;
  MEbinning           eta_binning_;
  MEbinning           HT_binning_;

  std::vector<double> met_variable_binning_;
  std::vector<double> HT_variable_binning_;
  std::vector<double> jetPt_variable_binning_;
  std::vector<double> muPt_variable_binning_;
  std::vector<double> elePt_variable_binning_;
  std::vector<double> jetEta_variable_binning_;
  std::vector<double> muEta_variable_binning_;
  std::vector<double> eleEta_variable_binning_;

  METME metME_;
  METME metME_variableBinning_;
  METME metVsLS_;
  METME metPhiME_;

  METME jetVsLS_;
  METME muVsLS_;
  METME eleVsLS_;
  METME htVsLS_;

  METME jetEtaPhi_; // for HEP17 monitoring


  METME jetMulti_;
  METME eleMulti_;
  METME muMulti_;

  std::vector<METME> muPhi_;
  std::vector<METME> muEta_;
  std::vector<METME> muPt_;

  std::vector<METME> elePhi_;
  std::vector<METME> eleEta_;
  std::vector<METME> elePt_;

  std::vector<METME> jetPhi_;
  std::vector<METME> jetEta_;
  std::vector<METME> jetPt_;

  std::vector<METME> muPt_variableBinning_;
  std::vector<METME> elePt_variableBinning_;
  std::vector<METME> jetPt_variableBinning_;

  std::vector<METME> muEta_variableBinning_;
  std::vector<METME> eleEta_variableBinning_;
  std::vector<METME> jetEta_variableBinning_;

  METME eventHT_;
  METME eventHT_variableBinning_;

  GenericTriggerEventFlag* num_genTriggerEventFlag_;
  GenericTriggerEventFlag* den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET,true>         metSelection_;
  StringCutObjectSelector<reco::PFJet,true   >    jetSelection_;
  StringCutObjectSelector<reco::GsfElectron,true> eleSelection_;
  StringCutObjectSelector<reco::Muon,true>        muoSelection_;
  StringCutObjectSelector<reco::PFJet,true   >    HTdefinition_;

  unsigned int njets_;
  unsigned int nelectrons_;
  unsigned int nmuons_;
  double leptJetDeltaRmin_;
  double HTcut_;

};

#endif // TOPMONITOR_H
