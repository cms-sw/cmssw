#ifndef HLTInclusiveVBFSource_H
#define HLTInclusiveVBFSource_H

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

//#include "RecoJets/JetProducers/interface/JetIDHelper.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class HLTInclusiveVBFSource : public DQMEDAnalyzer {
 public:
  explicit HLTInclusiveVBFSource(const edm::ParameterSet&);
  ~HLTInclusiveVBFSource();

  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
 private:
  virtual bool isBarrel(double eta);
  virtual bool isEndCap(double eta); 
  virtual bool isForward(double eta);
  virtual bool validPathHLT(std::string path);
  virtual bool isHLTPathAccepted(std::string pathName);
  virtual bool isTriggerObjectFound(std::string objectName);
  //virtual double TriggerPosition(std::string trigName);
  
  // ----------member data --------------------------- 
  int nCount_;
      
  std::vector<int>  prescUsed_;
  
  std::string dirname_;
  std::string processname_;
  //reco::helper::JetIDHelper *jetID; // JetID helper (Need to run with RECO, not AOD)
  std::vector<std::string> path_;
  
  bool debug_;

  double minPtHigh_;
  double minPtLow_;
  double minDeltaEta_;
  double minInvMass_;
  double deltaRMatch_;
  bool   etaOpposite_; 
  
  edm::InputTag triggerSummaryLabel_;
  edm::Handle<trigger::TriggerEvent> triggerObj_;
  edm::InputTag triggerResultsLabel_;
  edm::Handle<edm::TriggerResults> triggerResults_; 
  edm::TriggerNames triggerNames_; // TriggerNames class
  
  edm::EDGetTokenT <edm::TriggerResults> triggerResultsToken;
  edm::EDGetTokenT <edm::TriggerResults> triggerResultsFUToken;
  edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryToken;
  edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryFUToken;

  edm::EDGetTokenT <edm::View<reco::PFJet> > pfJetsToken;
  edm::EDGetTokenT <edm::View<reco::PFMET> > pfMetToken;
  edm::EDGetTokenT <reco::CaloJetCollection> caloJetsToken;
  edm::EDGetTokenT <reco::CaloMETCollection> caloMetToken;

  edm::Handle<reco::CaloJetCollection> calojetColl_;
  edm::Handle<reco::CaloMETCollection> calometColl_; 
  edm::Handle<reco::PFJetCollection>   pfjetColl_;
  edm::Handle<reco::PFMETCollection>   pfmetColl_; 
  
  reco::CaloJetCollection calojet; 
  reco::PFJetCollection pfjet; 
  HLTConfigProvider hltConfig_;

  bool check_mjj650_Pt35_DEta3p5;
  bool check_mjj700_Pt35_DEta3p5;
  bool check_mjj750_Pt35_DEta3p5;
  bool check_mjj800_Pt35_DEta3p5;
  bool check_mjj650_Pt40_DEta3p5; 
  bool check_mjj700_Pt40_DEta3p5;
  bool check_mjj750_Pt40_DEta3p5; 
  bool check_mjj800_Pt40_DEta3p5;
  
  std::string pathname;
  std::string filtername;

  double reco_ejet1; 
  //double reco_etjet1;
  double reco_pxjet1;
  double reco_pyjet1;
  double reco_pzjet1;
  double reco_ptjet1;
  double reco_etajet1;
  double reco_phijet1;
  //
  double reco_ejet2;
  //double reco_etjet2;
  double reco_pxjet2;
  double reco_pyjet2;
  double reco_pzjet2;
  double reco_ptjet2;
  double reco_etajet2;
  double reco_phijet2;
  //  
  double hlt_ejet1;
  //double hlt_etjet1;
  double hlt_pxjet1;
  double hlt_pyjet1;
  double hlt_pzjet1;
  double hlt_ptjet1;
  double hlt_etajet1;
  double hlt_phijet1;
  //
  double hlt_ejet2 ;
  //double hlt_etjet2;
  double hlt_pxjet2;
  double hlt_pyjet2;
  double hlt_pzjet2;
  double hlt_ptjet2;
  double hlt_etajet2;
  double hlt_phijet2;
  //
  bool checkOffline;
  bool checkHLT;
  bool checkHLTIndex;
  //
  float dR_HLT_RECO_11; 
  float dR_HLT_RECO_22;
  float dR_HLT_RECO_12; 
  float dR_HLT_RECO_21;
  bool checkdR_sameOrder;
  bool checkdR_crossOrder;
  //
  double reco_deltaetajet;
  double reco_deltaphijet;
  double reco_invmassjet;
  double hlt_deltaetajet;
  double hlt_deltaphijet;
  double hlt_invmassjet;
  
  // helper class to store the data path
  
  class PathInfo {
    PathInfo():
      prescaleUsed_(-1), 
      pathName_("unset"), 
      filterName_("unset"), 
      processName_("unset"), 
      objectType_(-1), 
      triggerType_("unset"){};
      //
  public:
    //
    void setHistos(
		   MonitorElement* const RECO_deltaEta_DiJet,
		   MonitorElement* const RECO_deltaPhi_DiJet,
		   MonitorElement* const RECO_invMass_DiJet,
		   MonitorElement* const HLT_deltaEta_DiJet,
		   MonitorElement* const HLT_deltaPhi_DiJet,
		   MonitorElement* const HLT_invMass_DiJet,
		   MonitorElement* const RECO_deltaEta_DiJet_Match,
		   MonitorElement* const RECO_deltaPhi_DiJet_Match,
		   MonitorElement* const RECO_invMass_DiJet_Match,
		   MonitorElement* const RECOHLT_deltaEta,
		   MonitorElement* const RECOHLT_deltaPhi,
		   MonitorElement* const RECOHLT_invMass,
		   MonitorElement* const NumberOfMatches,
		   MonitorElement* const NumberOfEvents
		   )    
    { 
      RECO_deltaEta_DiJet_       = RECO_deltaEta_DiJet;
      RECO_deltaPhi_DiJet_       = RECO_deltaPhi_DiJet;
      RECO_invMass_DiJet_        = RECO_invMass_DiJet;
      HLT_deltaEta_DiJet_        = HLT_deltaEta_DiJet;
      HLT_deltaPhi_DiJet_        = HLT_deltaPhi_DiJet ;
      HLT_invMass_DiJet_         = HLT_invMass_DiJet;
      RECO_deltaEta_DiJet_Match_ = RECO_deltaEta_DiJet_Match;
      RECO_deltaPhi_DiJet_Match_ = RECO_deltaPhi_DiJet_Match;
      RECO_invMass_DiJet_Match_  = RECO_invMass_DiJet_Match;
      RECOHLT_deltaEta_          = RECOHLT_deltaEta;
      RECOHLT_deltaPhi_          = RECOHLT_deltaPhi ;
      RECOHLT_invMass_           = RECOHLT_invMass;
      NumberOfMatches_           = NumberOfMatches;
      NumberOfEvents_            = NumberOfEvents;
    };
    ~PathInfo() {};
    PathInfo(int prescaleUsed, 
	     std::string pathName,
	     std::string filterName, 
	     std::string processName, 
	     size_t type, 
	     std::string triggerType):
      prescaleUsed_(prescaleUsed), 
      pathName_(pathName), 
      filterName_(filterName), 
      processName_(processName), 
      objectType_(type), 
      triggerType_(triggerType){};
    
      MonitorElement * getMEhisto_RECO_deltaEta_DiJet()       { return RECO_deltaEta_DiJet_; }
      MonitorElement * getMEhisto_RECO_deltaPhi_DiJet()       { return RECO_deltaPhi_DiJet_; }
      MonitorElement * getMEhisto_RECO_invMass_DiJet()        { return RECO_invMass_DiJet_; }
      MonitorElement * getMEhisto_HLT_deltaEta_DiJet()        { return HLT_deltaEta_DiJet_; }
      MonitorElement * getMEhisto_HLT_deltaPhi_DiJet()        { return HLT_deltaPhi_DiJet_; }
      MonitorElement * getMEhisto_HLT_invMass_DiJet()         { return HLT_invMass_DiJet_; }
      MonitorElement * getMEhisto_RECO_deltaEta_DiJet_Match() { return RECO_deltaEta_DiJet_Match_; }
      MonitorElement * getMEhisto_RECO_deltaPhi_DiJet_Match() { return RECO_deltaPhi_DiJet_Match_; }
      MonitorElement * getMEhisto_RECO_invMass_DiJet_Match()  { return RECO_invMass_DiJet_Match_; }
      MonitorElement * getMEhisto_RECOHLT_deltaEta()          { return RECOHLT_deltaEta_; }
      MonitorElement * getMEhisto_RECOHLT_deltaPhi()          { return RECOHLT_deltaPhi_; }
      MonitorElement * getMEhisto_RECOHLT_invMass()           { return RECOHLT_invMass_; }
      MonitorElement * getMEhisto_NumberOfMatches()           { return NumberOfMatches_; }
      MonitorElement * getMEhisto_NumberOfEvents()            { return NumberOfEvents_; }
      
      const std::string getLabel(void ) const {
	return filterName_;
      }
      void setLabel(std::string labelName){
	filterName_ = labelName;
	return;
      }
      const std::string getPath(void ) const {
	return pathName_;
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
      bool operator==(const std::string v){
	return v==pathName_;
      }
      
  private:
      int prescaleUsed_;
      std::string pathName_;
      std::string filterName_;
      std::string processName_;
      int objectType_;
      std::string triggerType_;

      MonitorElement*  RECO_deltaEta_DiJet_;
      MonitorElement*  RECO_deltaPhi_DiJet_;
      MonitorElement*  RECO_invMass_DiJet_;
      MonitorElement*  HLT_deltaEta_DiJet_;
      MonitorElement*  HLT_deltaPhi_DiJet_;
      MonitorElement*  HLT_invMass_DiJet_;
      MonitorElement*  RECO_deltaEta_DiJet_Match_;
      MonitorElement*  RECO_deltaPhi_DiJet_Match_;
      MonitorElement*  RECO_invMass_DiJet_Match_;
      MonitorElement*  RECOHLT_deltaEta_;
      MonitorElement*  RECOHLT_deltaPhi_;
      MonitorElement*  RECOHLT_invMass_;
      MonitorElement*  NumberOfMatches_;
      MonitorElement*  NumberOfEvents_;
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
  PathInfoCollection hltPathsAll_;
};
#endif
