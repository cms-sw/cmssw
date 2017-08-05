#ifndef DQMOffline_Trigger_TopMonitor_h
#define DQMOffline_Trigger_TopMonitor_h

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
#include "DataFormats/Math/interface/deltaR.h"
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
// Marina
#include "DataFormats/BTauReco/interface/JetTag.h"
//Suvankar
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


class GenericTriggerEventFlag;

struct MEbinning {
  unsigned int nbins;
  double xmin;
  double xmax;
};

struct METME {
  MonitorElement* numerator;
  MonitorElement* denominator;
};

//Suvankar
struct PVcut {
  double dxy;
  double dz;
};


//
// class declaration
//

class TopMonitor : public DQMEDAnalyzer 
{
public:
  TopMonitor( const edm::ParameterSet& );
  ~TopMonitor() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, METME& me, const std::string& histname, const std::string& histtitle, unsigned int nbins, double xmin, double xmax);
  void bookME(DQMStore::IBooker &, METME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX);
  void bookME(DQMStore::IBooker &, METME& me, const std::string& histname, const std::string& histtitle, unsigned int nbinsX, double xmin, double xmax, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, METME& me, const std::string& histname, const std::string& histtitle, unsigned int nbinsX, double xmin, double xmax, unsigned int nbinsY, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, METME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY);
  void setMETitle(METME& me, const std::string& titleX, const std::string& titleY);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

  // Marina
  struct JetRefCompare :
    public std::binary_function<edm::RefToBase<reco::Jet>, edm::RefToBase<reco::Jet>, bool> {
    inline bool operator () (const edm::RefToBase<reco::Jet> &j1, const edm::RefToBase<reco::Jet> &j2)
      const {return j1.id() < j2.id() || (j1.id() == j2.id() && j1.key() < j2.key());}
  };
  // Marina
  typedef std::map<edm::RefToBase<reco::Jet>, float, JetRefCompare> JetTagMap;



private:
  static MEbinning getHistoPSet    (const edm::ParameterSet& pset);
  static MEbinning getHistoLSPSet  (const edm::ParameterSet& pset);

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::PFMETCollection>       metToken_;
  edm::EDGetTokenT<reco::PFJetCollection>       jetToken_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron> > eleToken_;
  edm::EDGetTokenT<reco::MuonCollection>        muoToken_;
  // Marina
  edm::EDGetTokenT<reco::JetTagCollection>  jetTagToken_ ;
  //Suvankar
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;

  MEbinning           met_binning_;
  MEbinning           ls_binning_;
  MEbinning           phi_binning_;
  MEbinning           pt_binning_;
  MEbinning           eta_binning_;
  MEbinning           HT_binning_;
  MEbinning           DR_binning_;
  // Marina
  MEbinning           csv_binning_;
  //george
   MEbinning           invMass_mumu_binning_;
   MEbinning           MHT_binning_;


  std::vector<double> met_variable_binning_;
  std::vector<double> HT_variable_binning_;
  std::vector<double> jetPt_variable_binning_;
  std::vector<double> muPt_variable_binning_;
  std::vector<double> elePt_variable_binning_;
  std::vector<double> jetEta_variable_binning_;
  std::vector<double> muEta_variable_binning_;
  std::vector<double> eleEta_variable_binning_;
   //george
  std::vector<double> invMass_mumu_variable_binning_;
  std::vector<double> MHT_variable_binning_;

  std::vector<double> HT_variable_binning_2D_;
  std::vector<double> jetPt_variable_binning_2D_;
  std::vector<double> muPt_variable_binning_2D_;
  std::vector<double> elePt_variable_binning_2D_;
  std::vector<double> jetEta_variable_binning_2D_;
  std::vector<double> muEta_variable_binning_2D_;
  std::vector<double> eleEta_variable_binning_2D_;
  std::vector<double> phi_variable_binning_2D_;

  METME metME_;
  METME metME_variableBinning_;
  METME metVsLS_;
  METME metPhiME_;

  METME jetVsLS_;
  METME muVsLS_;
  METME eleVsLS_;
  // Marina
  METME bjetVsLS_;
  METME htVsLS_;

  METME jetEtaPhi_HEP17_; // for HEP17 monitoring

  METME jetMulti_;
  METME eleMulti_;
  METME muMulti_;
  // Marina
  METME bjetMulti_;

  METME elePt_jetPt_;
  METME elePt_eventHT_;

  METME ele1Pt_ele2Pt_;
  METME ele1Eta_ele2Eta_;
  METME mu1Pt_mu2Pt_;
  METME mu1Eta_mu2Eta_;
  METME elePt_muPt_;
  METME eleEta_muEta_;
  //george
  METME invMass_mumu_;
  METME eventMHT_;  
  METME invMass_mumu_variableBinning_;
  METME eventMHT_variableBinning_;

  //BTV
  METME DeltaR_jet_Mu_;

  std::vector<METME> muPhi_;
  std::vector<METME> muEta_;
  std::vector<METME> muPt_;

  std::vector<METME> elePhi_;
  std::vector<METME> eleEta_;
  std::vector<METME> elePt_;

  std::vector<METME> jetPhi_;
  std::vector<METME> jetEta_;
  std::vector<METME> jetPt_;

  // Marina
  std::vector<METME> bjetPhi_;
  std::vector<METME> bjetEta_;
  std::vector<METME> bjetPt_;
  std::vector<METME> bjetCSV_;
  
  std::vector<METME> muPt_variableBinning_;
  std::vector<METME> elePt_variableBinning_;
  std::vector<METME> jetPt_variableBinning_;
  // Marina
  std::vector<METME> bjetPt_variableBinning_;

  std::vector<METME> muEta_variableBinning_;
  std::vector<METME> eleEta_variableBinning_;
  std::vector<METME> jetEta_variableBinning_;
  // Marina
  std::vector<METME> bjetEta_variableBinning_;
  

  //2D distributions
  std::vector<METME> jetPtEta_;
  std::vector<METME> jetEtaPhi_;
  std::vector<METME> elePtEta_;
  std::vector<METME> eleEtaPhi_;
  std::vector<METME> muPtEta_;
  std::vector<METME> muEtaPhi_;
  // Marina
  std::vector<METME> bjetPtEta_;
  std::vector<METME> bjetEtaPhi_;
  std::vector<METME> bjetCSVHT_;

  METME eventHT_;
  METME eventHT_variableBinning_;
  

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET,true>         metSelection_;
  StringCutObjectSelector<reco::PFJet,true   >    jetSelection_;
  StringCutObjectSelector<reco::GsfElectron,true> eleSelection_;
  StringCutObjectSelector<reco::Muon,true>        muoSelection_;
  StringCutObjectSelector<reco::PFJet,true   >    HTdefinition_;
  
  //Suvankar
  StringCutObjectSelector<reco::Vertex,true>      vtxSelection_;
  
  StringCutObjectSelector<reco::Jet,true   >      bjetSelection_;

unsigned int njets_;
  unsigned int nelectrons_;
  unsigned int nmuons_;
  double leptJetDeltaRmin_;
  double bJetMuDeltaRmax_;
  double bJetDeltaEtaMax_;
  double HTcut_;
  // Marina
  unsigned int nbjets_;
  double workingpoint_;

  //Suvankar
  PVcut  lepPVcuts_;
  bool usePVcuts_;

  //george
  double invMassUppercut_;
  double invMassLowercut_;
  bool opsign_;
  StringCutObjectSelector<reco::PFJet,true   >    MHTdefinition_;
  double MHTcut_;
  double mll;
  int   sign;
  


  
};

#endif // DQMOffline_Trigger_TopMonitor_h
