/*
  New version of HLT Offline DQM code for BTag
  responsible: Jyothsna Komaragiri
*/

#ifndef BTagHLTOfflineSource_H
#define BTagHLTOfflineSource_H

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "RecoJets/JetProducers/interface/JetIDHelper.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include <iostream>
#include <fstream>
#include <vector>

class JetPtSorter {
public:
  template <class T> bool operator() ( const T& a, const T& b ) {
    return ( a.pt() > b.pt() );
  }
};

class BTagHLTOfflineSource : public edm::EDAnalyzer {

public:
  explicit BTagHLTOfflineSource(const edm::ParameterSet&);
  ~BTagHLTOfflineSource();
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  
  void histobooking( const edm::EventSetup& c);
  
  // EndRun
  void endRun(const edm::Run& run, const edm::EventSetup& c);
  
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			    const edm::EventSetup& c) ;
  
  // DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			  const edm::EventSetup& c);
  
  virtual bool isBarrel(double eta);
  virtual bool isEndCap(double eta); 
  virtual bool isForward(double eta);
  virtual bool validPathHLT(std::string path);
  virtual bool isHLTPathAccepted(std::string pathName);
  virtual bool isTriggerObjectFound(std::string objectName);
  virtual double TriggerPosition(std::string trigName);

  //Muon and Jet selectors
  void selectMuons(const edm::Handle<reco::MuonCollection> & muonHandle);
  bool isVBTFMuon(const reco::Muon& muon);
  void selectJets(const edm::Event& iEvent, const edm::Handle<reco::CaloJetCollection> & jetHandle);
  bool isMuonJet(const reco::CaloJet& calojet, reco::MuonCollection *SelectedMuons);

  virtual void fillMEforMonTriggerSummary();
  
  virtual void fillMEforEffAllTrigger(const edm::Event & iEvent);
  virtual void fillMEforEffWrtMuTrigger(const edm::Event & iEvent);
  virtual void fillMEforEffWrtMBTrigger(const edm::Event & iEvent);

  virtual void fillMEforTriggerNTfired();

  // ----------member data --------------------------- 
  int nev_;
  DQMStore * dbe;
  
  MonitorElement* total_;
  
  std::vector<std::string>  MuonTrigPaths_;
  std::vector<std::string>  MBTrigPaths_;
  std::vector<int>          prescUsed_;
  
  std::string dirname_;
  std::string processname_;

  // JetID helper
  reco::helper::JetIDHelper *jetID;
  
  bool verbose_;
  bool plotEff_ ; 
  bool isSetup_;
  bool nameForEff_;  

  //////////////////////////
  // Jet selection cuts
  //////////////////////////
  double _jetpt;
  double _jeteta;
  double _fEMF;
  double _fHPD;
  double _n90Hits; 

  // Muon selection cuts
  //////////////////////////
  double _mupt;
  double _mueta;
  int    _muonHits;
  int    _nMatches;
  int    _trackerHits;
  int    _pixelHits;
  int    _outerHits;
  double _tknormalizedChi2;
  double _gmnormalizedChi2;
  double _mudZ;
  double _mujetdR;

  //////////////////////////

  edm::InputTag triggerSummaryLabel_;
  edm::InputTag triggerResultsLabel_;

  // Input tags for calo jets and muons
  edm::InputTag caloJetsTag_;
  edm::InputTag muonTag_;

  // Selected for calo jets and muons
  reco::CaloJetCollection *SelectedCaloJets; 
  reco::MuonCollection    *SelectedMuons;

  edm::Handle<reco::CaloJetCollection>  SelectedCaloJetsColl_; 
  edm::Handle<reco::MuonCollection>     SelectedMuonsColl_;

  std::vector<std::string> custompathname;
  std::vector<std::pair<std::string, std::string> > custompathnamepairs_;

  HLTConfigProvider hltConfig_;
  edm::Handle<edm::TriggerResults> triggerResults_;
  edm::TriggerNames triggerNames_; // TriggerNames class
  edm::Handle<trigger::TriggerEvent> triggerObj_;
  edm::Handle<reco::BeamSpot> beamSpot_;

  // data across paths
  MonitorElement* scalersSelect;

  // helper class to store the data path
  class PathInfo {
    PathInfo():
      pathIndex_(-1), prescaleUsed_(-1),denomPathName_("unset"), pathName_("unset"), l1pathName_("unset"), filterName_("unset"), DenomfilterName_("unset"), processName_("unset"), objectType_(-1), triggerType_("unset")
    {};
  public:
    void setHistos(MonitorElement* const N, 
		   MonitorElement* const Pt, 
		   MonitorElement* const PtBarrel,
		   MonitorElement* const PtEndcap,
		   MonitorElement* const PtForward,
		   MonitorElement* const Eta,
		   MonitorElement* const Phi, 
		   MonitorElement* const EtaPhi, 
		   MonitorElement* const N_L1,
		   MonitorElement* const Pt_L1,
		   MonitorElement* const PtBarrel_L1,
		   MonitorElement* const PtEndcap_L1,
		   MonitorElement* const PtForward_L1,
		   MonitorElement* const Eta_L1,
		   MonitorElement* const Phi_L1,
		   MonitorElement* const EtaPhi_L1,
		   MonitorElement* const N_HLT,
		   MonitorElement* const Pt_HLT,
		   MonitorElement* const PtBarrel_HLT,
		   MonitorElement* const PtEndcap_HLT,
		   MonitorElement* const PtForward_HLT,
		   MonitorElement* const Eta_HLT,
		   MonitorElement* const Phi_HLT,
		   MonitorElement* const EtaPhi_HLT
		    )

          {
	    N_             = N;
	    Pt_            = Pt;
	    PtBarrel_      = PtBarrel;
	    PtEndcap_      = PtEndcap;
	    PtForward_     = PtForward;
	    Eta_           = Eta;
	    Phi_           = Phi;
	    EtaPhi_        = EtaPhi;
	    N_L1_          = N_L1;
	    Pt_L1_         = Pt_L1;
	    PtBarrel_L1_   = PtBarrel_L1;
	    PtEndcap_L1_   = PtEndcap_L1;
	    PtForward_L1_  = PtForward_L1;
	    Eta_L1_        = Eta_L1;
	    Phi_L1_        = Phi_L1;
	    EtaPhi_L1_     = EtaPhi_L1 ;
	    N_HLT_         = N_HLT;
	    Pt_HLT_        = Pt_HLT;
	    PtBarrel_HLT_  = PtBarrel_HLT;
	    PtEndcap_HLT_  = PtEndcap_HLT;
	    PtForward_HLT_ = PtForward_HLT;
	    Eta_HLT_       = Eta_HLT;
	    Phi_HLT_       = Phi_HLT;
	    EtaPhi_HLT_    = EtaPhi_HLT ;
	  };

    void setDgnsHistos(MonitorElement* const TriggerSummary,
                       MonitorElement* const JetSize,
                       MonitorElement* const JetPt,  
                       MonitorElement* const EtavsPt,
                       MonitorElement* const PhivsPt,
                       MonitorElement* const Pt12,
                       MonitorElement* const Eta12,
                       MonitorElement* const Phi12,                      
                       MonitorElement* const Pt3,
                       MonitorElement* const Pt12Pt3,
                       MonitorElement* const Pt12Phi12
		       )
    {
      TriggerSummary_ = TriggerSummary; 
      JetSize_      = JetSize;
      JetPt_        = JetPt;
      EtavsPt_      = EtavsPt;
      PhivsPt_      = PhivsPt;
      Pt12_         = Pt12;
      Eta12_        = Eta12;
      Phi12_        = Phi12;
      Pt3_          = Pt3;
      Pt12Pt3_      = Pt12Pt3;
      Pt12Phi12_    = Pt12Phi12; 
    };

    void setEffHistos( MonitorElement* const NumeratorPt,
                       MonitorElement* const NumeratorPtBarrel,
                       MonitorElement* const NumeratorPtEndcap,
                       MonitorElement* const NumeratorPtForward,
                       MonitorElement* const NumeratorEta,
                       MonitorElement* const NumeratorPhi,
                       MonitorElement* const NumeratorEtaPhi,
                       MonitorElement* const DenominatorPt,
                       MonitorElement* const DenominatorPtBarrel,
                       MonitorElement* const DenominatorPtEndcap,
                       MonitorElement* const DenominatorPtForward,
                       MonitorElement* const DenominatorEta,
                       MonitorElement* const DenominatorPhi,
                       MonitorElement* const DenominatorEtaPhi
		       )
    {
      NumeratorPt_            = NumeratorPt;
      NumeratorPtBarrel_      = NumeratorPtBarrel;
      NumeratorPtEndcap_      = NumeratorPtEndcap;
      NumeratorPtForward_     = NumeratorPtForward;
      NumeratorEta_           = NumeratorEta;
      NumeratorPhi_           = NumeratorPhi;
      NumeratorEtaPhi_        = NumeratorEtaPhi; 
      DenominatorPt_          = DenominatorPt;
      DenominatorPtBarrel_    = DenominatorPtBarrel;
      DenominatorPtEndcap_    = DenominatorPtEndcap;
      DenominatorPtForward_   = DenominatorPtForward;
      DenominatorEta_         = DenominatorEta;
      DenominatorPhi_         = DenominatorPhi;
      DenominatorEtaPhi_      = DenominatorEtaPhi;
    }; 

    ~PathInfo() {};
    PathInfo(int prescaleUsed, std::string denomPathName, std::string pathName, std::string l1pathName, std::string filterName, std::string DenomfilterName, std::string processName, size_t type, std::string triggerType):
      prescaleUsed_(prescaleUsed),denomPathName_(denomPathName), pathName_(pathName), l1pathName_(l1pathName), filterName_(filterName), DenomfilterName_(DenomfilterName), processName_(processName), objectType_(type), triggerType_(triggerType){};
    
    MonitorElement * getMEhisto_N() { return N_;}
    MonitorElement * getMEhisto_Pt() { return Pt_;}
    MonitorElement * getMEhisto_PtBarrel() { return PtBarrel_;}
    MonitorElement * getMEhisto_PtEndcap() { return PtEndcap_;}
    MonitorElement * getMEhisto_PtForward() { return PtForward_;}
    MonitorElement * getMEhisto_Eta() { return Eta_; }
    MonitorElement * getMEhisto_Phi() { return Phi_; }
    MonitorElement * getMEhisto_EtaPhi() { return EtaPhi_; }
    
    MonitorElement * getMEhisto_N_L1() { return N_L1_;}
    MonitorElement * getMEhisto_Pt_L1() { return Pt_L1_;}
    MonitorElement * getMEhisto_PtBarrel_L1() { return PtBarrel_L1_;}
    MonitorElement * getMEhisto_PtEndcap_L1() { return PtEndcap_L1_;}
    MonitorElement * getMEhisto_PtForward_L1() { return PtForward_L1_;}
    MonitorElement * getMEhisto_Eta_L1() { return Eta_L1_; }
    MonitorElement * getMEhisto_Phi_L1() { return Phi_L1_; }
    MonitorElement * getMEhisto_EtaPhi_L1() { return EtaPhi_L1_; } 
    
    MonitorElement * getMEhisto_N_HLT() { return N_HLT_;}
    MonitorElement * getMEhisto_Pt_HLT() { return Pt_HLT_;}
    MonitorElement * getMEhisto_PtBarrel_HLT() { return PtBarrel_HLT_;}
    MonitorElement * getMEhisto_PtEndcap_HLT() { return PtEndcap_HLT_;}
    MonitorElement * getMEhisto_PtForward_HLT() { return PtForward_HLT_;}
    MonitorElement * getMEhisto_Eta_HLT() { return Eta_HLT_; }
    MonitorElement * getMEhisto_Phi_HLT() { return Phi_HLT_; }
    MonitorElement * getMEhisto_EtaPhi_HLT() { return EtaPhi_HLT_; }
    
    MonitorElement * getMEhisto_NumeratorPt() { return NumeratorPt_;}
    MonitorElement * getMEhisto_NumeratorPtBarrel() { return NumeratorPtBarrel_;}
    MonitorElement * getMEhisto_NumeratorPtEndcap() { return NumeratorPtEndcap_;}
    MonitorElement * getMEhisto_NumeratorPtForward() { return NumeratorPtForward_;}
    MonitorElement * getMEhisto_NumeratorEta() { return NumeratorEta_; }
    MonitorElement * getMEhisto_NumeratorPhi() { return NumeratorPhi_; }
    MonitorElement * getMEhisto_NumeratorEtaPhi() { return NumeratorEtaPhi_; } 
    MonitorElement * getMEhisto_DenominatorPt() { return DenominatorPt_;}
    MonitorElement * getMEhisto_DenominatorPtBarrel() { return DenominatorPtBarrel_;}
    MonitorElement * getMEhisto_DenominatorPtEndcap() { return DenominatorPtEndcap_;}
    MonitorElement * getMEhisto_DenominatorPtForward() { return DenominatorPtForward_;}
    MonitorElement * getMEhisto_DenominatorEta() { return DenominatorEta_; }
    MonitorElement * getMEhisto_DenominatorPhi() { return DenominatorPhi_; }
    MonitorElement * getMEhisto_DenominatorEtaPhi() { return DenominatorEtaPhi_; }
    
    MonitorElement * getMEhisto_TriggerSummary() {return TriggerSummary_;}
    MonitorElement * getMEhisto_JetSize() {return JetSize_;}
    MonitorElement * getMEhisto_JetPt() {return JetPt_;}
    MonitorElement * getMEhisto_EtavsPt(){return EtavsPt_;}
    MonitorElement * getMEhisto_PhivsPt(){return PhivsPt_;}
    MonitorElement * getMEhisto_Pt12() {return Pt12_;}
    MonitorElement * getMEhisto_Eta12() {return Eta12_;}
    MonitorElement * getMEhisto_Phi12() {return Phi12_;}   
    MonitorElement * getMEhisto_Pt3() {return Pt3_;}
    MonitorElement * getMEhisto_Pt12Pt3() {return Pt12Pt3_;}
    MonitorElement * getMEhisto_Pt12Phi12() {return Pt12Phi12_;}
    
    const std::string getLabel(void ) const {
      return filterName_;
    }
    const std::string getDenomLabel(void ) const {
      return DenomfilterName_;
    }
    
    void setLabel(std::string labelName){
      filterName_ = labelName;
      return;
    }
    void setDenomLabel(std::string labelName){
      DenomfilterName_ = labelName;
      return;
    }
    const std::string getPath(void ) const {
      return pathName_;
    }
    const std::string getl1Path(void ) const {
      return l1pathName_;
    }
    const std::string getDenomPath(void ) const {
      return denomPathName_;
    }
    const int getprescaleUsed(void) const {
      return prescaleUsed_;
    }
    const std::string getProcess(void ) const {
      return processName_;
    }
    const int getObjectType(void ) const {
      return objectType_;
    }
    const std::string getTriggerType(void ) const {
      return triggerType_;
    }    
    
    const edm::InputTag getTag(void) const{
      edm::InputTag tagName(filterName_,"",processName_);
      return tagName;
    }
    const edm::InputTag getDenomTag(void) const{
      edm::InputTag tagName(DenomfilterName_,"",processName_);
      return tagName;
    }
    bool operator==(const std::string v)
    {
      return v==pathName_;
    } 

  private:
    int pathIndex_;
    int prescaleUsed_;
    std::string denomPathName_;
    std::string pathName_;
    std::string l1pathName_;
    std::string filterName_;
    std::string DenomfilterName_;
    std::string processName_;
    int objectType_;
    std::string triggerType_;
    
    MonitorElement*  N_;
    MonitorElement*  Pt_;
    MonitorElement*  PtBarrel_;
    MonitorElement*  PtEndcap_;
    MonitorElement*  PtForward_;
    MonitorElement*  Eta_;
    MonitorElement*  Phi_;
    MonitorElement*  EtaPhi_;
    MonitorElement*  N_L1_;
    MonitorElement*  Pt_L1_;
    MonitorElement*  PtBarrel_L1_;
    MonitorElement*  PtEndcap_L1_;
    MonitorElement*  PtForward_L1_;
    MonitorElement*  Eta_L1_;
    MonitorElement*  Phi_L1_;
    MonitorElement*  EtaPhi_L1_;
    MonitorElement*  N_HLT_;
    MonitorElement*  Pt_HLT_;
    MonitorElement*  PtBarrel_HLT_;
    MonitorElement*  PtEndcap_HLT_;
    MonitorElement*  PtForward_HLT_;
    MonitorElement*  Eta_HLT_;
    MonitorElement*  Phi_HLT_;
    MonitorElement*  EtaPhi_HLT_;
    
    MonitorElement*  NumeratorPt_;
    MonitorElement*  NumeratorPtBarrel_;
    MonitorElement*  NumeratorPtEndcap_;
    MonitorElement*  NumeratorPtForward_;
    MonitorElement*  NumeratorEta_;
    MonitorElement*  NumeratorPhi_;
    MonitorElement*  NumeratorEtaPhi_;
    MonitorElement*  DenominatorPt_;
    MonitorElement*  DenominatorPtBarrel_;
    MonitorElement*  DenominatorPtEndcap_;
    MonitorElement*  DenominatorPtForward_;
    MonitorElement*  DenominatorEta_;
    MonitorElement*  DenominatorPhi_;
    MonitorElement*  DenominatorEtaPhi_;
    
    MonitorElement*  TriggerSummary_;
    MonitorElement*  JetSize_;
    MonitorElement*  JetPt_;
    MonitorElement*  EtavsPt_;
    MonitorElement*  PhivsPt_;
    MonitorElement*  Pt12_;
    MonitorElement*  Eta12_;
    MonitorElement*  Phi12_; 
    MonitorElement*  Pt3_;
    MonitorElement*  Pt12Pt3_;
    MonitorElement*  Pt12Phi12_;
  };
  
  // simple collection 
  class PathInfoCollection: public std::vector<PathInfo> {
  public:
    PathInfoCollection(): std::vector<PathInfo>()
    {};
    std::vector<PathInfo>::iterator find(std::string pathName) {
      return std::find(begin(), end(), pathName);
    }
  };
  PathInfoCollection hltPathsAllTriggerSummary_;
  PathInfoCollection hltPathsAll_;
  PathInfoCollection hltPathsAllWrtMu_;
  PathInfoCollection hltPathsEff_;
  PathInfoCollection hltPathsEffWrtMu_;
  PathInfoCollection hltPathsEffWrtMB_;
  
  MonitorElement* rate_All;
  MonitorElement* rate_AllWrtMu;
  MonitorElement* rate_AllWrtMB;
  
  MonitorElement* correlation_All;
  MonitorElement* correlation_AllWrtMu;
  MonitorElement* correlation_AllWrtMB;
  
};
#endif

