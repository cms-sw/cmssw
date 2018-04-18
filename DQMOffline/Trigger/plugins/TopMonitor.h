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
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"

//ATHER                                                                                                                                                                                                            
#include "DataFormats/Common/interface/ValueMap.h"

class GenericTriggerEventFlag;


//
// class declaration
//

class TopMonitor : public DQMEDAnalyzer, public TriggerDQMBase
{
public:
  TopMonitor( const edm::ParameterSet& );
  ~TopMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
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

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::PFMETCollection>         metToken_;
  edm::EDGetTokenT<reco::PFJetCollection>         jetToken_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron> > eleToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> >          elecIDToken_; //ATHER   
  edm::EDGetTokenT<reco::MuonCollection>          muoToken_;
  edm::EDGetTokenT<reco::PhotonCollection>        phoToken_;
  // Marina
  edm::EDGetTokenT<reco::JetTagCollection>        jetTagToken_ ;
  //Suvankar
  edm::EDGetTokenT<reco::VertexCollection>        vtxToken_;

 //Suvankar
  struct PVcut {
      double dxy;
      double dz;
  };

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
  std::vector<double> phoPt_variable_binning_2D_;
  std::vector<double> jetEta_variable_binning_2D_;
  std::vector<double> muEta_variable_binning_2D_;
  std::vector<double> eleEta_variable_binning_2D_;
  std::vector<double> phoEta_variable_binning_2D_;
  std::vector<double> phi_variable_binning_2D_;

  ObjME metME_;
  ObjME metME_variableBinning_;
  ObjME metVsLS_;
  ObjME metPhiME_;

  ObjME jetVsLS_;
  ObjME muVsLS_;
  ObjME eleVsLS_;
	//Menglei
  ObjME phoVsLS_;
  // Marina
  ObjME bjetVsLS_;
  ObjME htVsLS_;

  ObjME jetEtaPhi_HEP17_; // for HEP17 monitoring

  ObjME jetMulti_;
  ObjME eleMulti_;
  ObjME muMulti_;
	//Menglei
  ObjME phoMulti_;
  // Marina
  ObjME bjetMulti_;

  ObjME elePt_jetPt_;
  ObjME elePt_eventHT_;

  ObjME ele1Pt_ele2Pt_;
  ObjME ele1Eta_ele2Eta_;
  ObjME mu1Pt_mu2Pt_;
  ObjME mu1Eta_mu2Eta_;
  ObjME elePt_muPt_;
  ObjME eleEta_muEta_;
  //george
  ObjME invMass_mumu_;
  ObjME eventMHT_;  
  ObjME invMass_mumu_variableBinning_;
  ObjME eventMHT_variableBinning_;
	//Menglei
  ObjME muPt_phoPt_;
  ObjME muEta_phoEta_;

  //BTV
  ObjME DeltaR_jet_Mu_;

  ObjME eventHT_;
  ObjME eventHT_variableBinning_;
  
  std::vector<ObjME> muPhi_;
  std::vector<ObjME> muEta_;
  std::vector<ObjME> muPt_;

  std::vector<ObjME> elePhi_;
  std::vector<ObjME> eleEta_;
  std::vector<ObjME> elePt_;

  std::vector<ObjME> jetPhi_;
  std::vector<ObjME> jetEta_;
  std::vector<ObjME> jetPt_;

  std::vector<ObjME> phoPhi_;
  std::vector<ObjME> phoEta_;
  std::vector<ObjME> phoPt_;


  // Marina
  std::vector<ObjME> bjetPhi_;
  std::vector<ObjME> bjetEta_;
  std::vector<ObjME> bjetPt_;
  std::vector<ObjME> bjetCSV_;
  
  std::vector<ObjME> muPt_variableBinning_;
  std::vector<ObjME> elePt_variableBinning_;
  std::vector<ObjME> jetPt_variableBinning_;
  // Marina
  std::vector<ObjME> bjetPt_variableBinning_;

  std::vector<ObjME> muEta_variableBinning_;
  std::vector<ObjME> eleEta_variableBinning_;
  std::vector<ObjME> jetEta_variableBinning_;
  // Marina
  std::vector<ObjME> bjetEta_variableBinning_;
  

  //2D distributions
  std::vector<ObjME> jetPtEta_;
  std::vector<ObjME> jetEtaPhi_;
  std::vector<ObjME> elePtEta_;
  std::vector<ObjME> eleEtaPhi_;
  std::vector<ObjME> muPtEta_;
  std::vector<ObjME> muEtaPhi_;
	//Menglei
  std::vector<ObjME> phoPtEta_;
  std::vector<ObjME> phoEtaPhi_;
  // Marina
  std::vector<ObjME> bjetPtEta_;
  std::vector<ObjME> bjetEtaPhi_;
  std::vector<ObjME> bjetCSVHT_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET,true>         metSelection_;
  StringCutObjectSelector<reco::PFJet,true   >    jetSelection_;
  StringCutObjectSelector<reco::GsfElectron,true> eleSelection_;
  StringCutObjectSelector<reco::Muon,true>        muoSelection_;
  StringCutObjectSelector<reco::Photon, true>     phoSelection_;
  StringCutObjectSelector<reco::PFJet,true   >    HTdefinition_;
  
  //Suvankar
  StringCutObjectSelector<reco::Vertex,true>      vtxSelection_;
  
  StringCutObjectSelector<reco::Jet,true   >      bjetSelection_;

  unsigned int njets_;
  unsigned int nelectrons_;
  unsigned int nmuons_;
  unsigned int nphotons_;
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

  bool applyMETcut_ = false;

  //george
  double invMassUppercut_;
  double invMassLowercut_;
  bool opsign_;
  StringCutObjectSelector<reco::PFJet,true   >    MHTdefinition_;
  double MHTcut_;
  double mll;
  int   sign;
bool invMassCutInAllMuPairs_;  

  //Menglei
  bool enablePhotonPlot_;

  //Mateusz
  bool enableMETplot_;
  
};

#endif // DQMOffline_Trigger_TopMonitor_h
