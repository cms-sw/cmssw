#ifndef HLTriggerOffline_Egamma_EmCheckEfficiency_H
#define HLTriggerOffline_Egamma_EmCheckEfficiency_H


// Base Class Headers
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include <vector>
#include "TDirectory.h"
#include "TH1F.h"
#include "TH2F.h"
#include "HepMC/GenParticle.h"

class EmCheckEfficiency : public edm::EDAnalyzer{
public:
  /// Constructor
  explicit EmCheckEfficiency(const edm::ParameterSet& pset);

  /// Destructor
  ~EmCheckEfficiency();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup&);
  void beginJob(const edm::EventSetup&);
  void endJob();

private:
  // Input from cfg file
  std::vector<edm::InputTag> theHLTCollectionLabels;  
  edm::InputTag theL1Seed;
  std::vector<int> theHLTOutputTypes;
  std::vector<bool> plotiso;
  std::vector<std::vector<edm::InputTag> > isoNames; // there has to be a better solution
  std::vector<std::pair<double,double> > plotBounds; 
  std::string theHltName;
  unsigned int reqNum;
  double thePtMin ;
  double thePtMax ;
  unsigned int theNbins ;

  std::vector<TH1F*> etahist;
  std::vector<TH1F*> ethist;
  std::vector<TH1F*> etahistmatch;
  std::vector<TH1F*> ethistmatch;
  std::vector<TH1F*> etahistoff;
  std::vector<TH1F*> ethistoff;
  std::vector<TH2F*> etahistiso;
  std::vector<TH2F*> ethistiso;
  TH1F* total;
  TH1F* totaloff;
  // TH1F* deta;
  ///TH1F* dphi;
   // TH1F* deta;
  ///TH1F* dphi;
  TH1F* etgen;
  TH1F* etagen;
  TH1F* etoff;
  TH1F* etaoff;
  int   pdgGen;
  double genEtaAcc;
  double genEtAcc;
  bool _doMC;
  bool _doOffline;

  template <class T> void fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& ,const edm::Event& ,unsigned int, std::vector<HepMC::GenParticle>& ,reco::GsfElectronCollection& );
  

};
#endif
