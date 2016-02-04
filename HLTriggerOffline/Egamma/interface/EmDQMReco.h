#ifndef HLTriggerOffline_Egamma_EmDQMReco_H
#define HLTriggerOffline_Egamma_EmDQMReco_H


// Base Class Headers
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include <vector>
#include "TDirectory.h"
#include "HepMC/GenParticle.h"
#include "CommonTools/Utils/interface/PtComparator.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class EmDQMReco : public edm::EDAnalyzer{
public:
  /// Constructor
  explicit EmDQMReco(const edm::ParameterSet& pset);

  /// Destructor
  ~EmDQMReco();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup&);
  void beginJob();
  void endJob();
  void beginRun( const edm::Run&, const edm::EventSetup& );
 
private:
  // Input from cfg file
  std::vector<edm::InputTag> theHLTCollectionLabels;  
  unsigned int numOfHLTCollectionLabels;  // Will be size of above vector
  bool useHumanReadableHistTitles;
  std::vector<std::string> theHLTCollectionHumanNames; // Human-readable names for the collections
  edm::InputTag theL1Seed;
  std::vector<int> theHLTOutputTypes;
  std::vector<bool> plotiso;
  std::vector<std::vector<edm::InputTag> > isoNames; // there has to be a better solution
  std::vector<std::pair<double,double> > plotBounds; 
  std::string theHltName;
  HLTConfigProvider hltConfig_;
  bool isHltConfigInitialized_;

  ////////////////////////////////////////////////////////////
  //          Read from configuration file                  //
  ////////////////////////////////////////////////////////////
  // paramters for generator study
  unsigned int reqNum;
  int   pdgGen;
  double recoEtaAcc;
  double recoEtAcc;
  // plotting paramters
  double plotEtaMax;
  double plotPtMin ;
  double plotPtMax ;
  unsigned int plotBins ;
  // preselction cuts
  edm::InputTag recocutCollection_;
  unsigned int recocut_;


  ////////////////////////////////////////////////////////////
  //          Create Histograms                             //
  ////////////////////////////////////////////////////////////
  // Et & eta distributions (RECO)
  std::vector<MonitorElement*> etahist;
  std::vector<MonitorElement*> ethist;
  std::vector<MonitorElement*> etahistmatchreco;
  std::vector<MonitorElement*> ethistmatchreco;
  std::vector<MonitorElement*> etahistmatchrecomonpath;
  std::vector<MonitorElement*> ethistmatchrecomonpath;
  std::vector<MonitorElement*> histEtOfHltObjMatchToReco;
  std::vector<MonitorElement*> histEtaOfHltObjMatchToReco;
  // Isolation distributions 
  std::vector<MonitorElement*> etahistiso;
  std::vector<MonitorElement*> ethistiso;
  std::vector<MonitorElement*> etahistisomatchreco;
  std::vector<MonitorElement*> ethistisomatchreco;
  std::vector<MonitorElement*> histEtIsoOfHltObjMatchToReco;
  std::vector<MonitorElement*> histEtaIsoOfHltObjMatchToReco;
  // Plots of efficiency per step
  MonitorElement* totalreco;
  MonitorElement* totalmatchreco;
  //reco histograms
  MonitorElement* etreco;
  MonitorElement* etareco;
  MonitorElement* etahistmonpath;
  MonitorElement* ethistmonpath;
  MonitorElement* etrecomonpath;
  MonitorElement* etarecomonpath;

  int eventnum;
  // int prescale;

  // interface to DQM framework
  DQMStore * dbe;
  std::string dirname_;

  template <class T> void fillHistos(edm::Handle<trigger::TriggerEventWithRefs>&,const edm::Event& ,unsigned int, std::vector<reco::Particle>&, bool, bool);
  GreaterByPt<reco::Particle> pTComparator_;
  GreaterByPt<reco::GsfElectron> pTRecoComparator_;
  
};
#endif
