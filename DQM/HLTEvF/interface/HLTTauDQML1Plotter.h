/* HLTTau Path  Analyzer
Michail Bachtis
University of Wisconsin - Madison
bachtis@hep.wisc.edu
*/



#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
//Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

typedef math::XYZTLorentzVectorD   LV;
typedef std::vector<LV>            LVColl;


class HLTTauDQML1Plotter  {
  
 public:
   HLTTauDQML1Plotter(const edm::ParameterSet&,int,int,int,double,bool,double);
  ~HLTTauDQML1Plotter();
  void analyze(const edm::Event&, const edm::EventSetup&, const std::vector<LVColl>&);
  
 private:
  void endJob() ;

  //helper functions
  std::pair<bool,LV> match(const LV&,const LVColl&,double);


  //Just a tag for better file organization
  std::string triggerTag_;

  //The  filters
  edm::InputTag l1ExtraTaus_;
  edm::InputTag l1ExtraJets_;
  edm::InputTag l1ExtraLeptons_;
  
  int LeptonType_;

  //Parameters(Note that the first entry is for the reference events)
  unsigned nTriggeredTaus_;
  unsigned nTriggeredLeptons_;

  bool doRefAnalysis_;
  double matchDeltaR_;


  double maxEt_;
  int binsEt_;
  int binsEta_;
  int binsPhi_;



  //MonitorElements general
  MonitorElement* l1tauEt_;
  MonitorElement* l1tauEta_;
  MonitorElement* l1tauPhi_;

  MonitorElement* l1jetEt_;
  MonitorElement* l1jetEta_;
  MonitorElement* l1jetPhi_;

  MonitorElement* l1leptonEt_;
  MonitorElement* l1leptonEta_;
  MonitorElement* l1leptonPhi_;


  
  //Monitor Elements for matching
  MonitorElement* inputEvents_;
  MonitorElement* l1tauEtRes_;

  MonitorElement* l1tauEtEffNum_;
  MonitorElement* l1tauEtEffDenom_;

  MonitorElement* l1tauEtaEffNum_;
  MonitorElement* l1tauEtaEffDenom_;

  MonitorElement* l1tauPhiEffNum_;
  MonitorElement* l1tauPhiEffDenom_;

  MonitorElement* l1jetEtEffNum_;
  MonitorElement* l1jetEtEffDenom_;

  MonitorElement* l1jetEtaEffNum_;
  MonitorElement* l1jetEtaEffDenom_;

  MonitorElement* l1jetPhiEffNum_;
  MonitorElement* l1jetPhiEffDenom_;

  MonitorElement* l1leptonEtEffNum_;
  MonitorElement* l1leptonEtEffDenom_;

  MonitorElement* l1leptonEtaEffNum_;
  MonitorElement* l1leptonEtaEffDenom_;

  MonitorElement* l1leptonPhiEffNum_;
  MonitorElement* l1leptonPhiEffDenom_;

  MonitorElement* l1tauPath_;


struct ComparePt
{
  bool operator()(LV l1,LV l2)
  {
    return l1.pt() > l1.pt() ;

  }
};  

  ComparePt ptSort;



};

