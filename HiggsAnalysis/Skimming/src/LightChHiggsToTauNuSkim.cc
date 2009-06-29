
/** \class LightChHiggsToTauNuSkim
 *
 * Consult header file for description
 *
 * \author:  Nuno Almeida - LIP-Lisbon
 */
 



#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HiggsAnalysis/Skimming/interface/LightChHiggsToTauNuSkim.h"



// Muons
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"


// Electrons
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

// Jets
#include "DataFormats/JetReco/interface/Jet.h"



using namespace std;
using namespace edm;
using namespace reco;



// Constructor
LightChHiggsToTauNuSkim::LightChHiggsToTauNuSkim(const edm::ParameterSet& cfg) :

  jetsTag_      (cfg.getParameter<InputTag>("jetsTag")),
  muonsTag_     (cfg.getParameter<InputTag>("muonsTag")),
  electronsTag_ (cfg.getParameter<InputTag>("electronsTag")),

  minNumbOfjets_(cfg.getParameter<int>("minNumbOfJets")),
 
  jetPtMin_    (cfg.getParameter<double>("jetPtMin")),
  jetEtaMin_    (cfg.getParameter<double>("jetEtaMin")),
  jetEtaMax_    (cfg.getParameter<double>("jetEtaMax")),

  
  leptonPtMin_  (cfg.getParameter<double>("leptonPtMin")),
  leptonEtaMin_ (cfg.getParameter<double>("leptonEtaMin")),
  leptonEtaMax_ (cfg.getParameter<double>("leptonEtaMax"))

{
  nEvents_         = 0;
  nSelectedEvents_ = 0;

}

// Destructor
LightChHiggsToTauNuSkim::~LightChHiggsToTauNuSkim() {

  cout << "LightChHiggsToTauNuSkim: \n" 
  << " N_events_HLTread= "  << nEvents_          
  << " N_events_Skimkept= " << nSelectedEvents_ ;
  if(nEvents_){ cout << " RelEfficiencyFilter= " << double(nSelectedEvents_)/double(nEvents_) << endl;}
  else { cout << " RelEfficiencyFilter= 0"  << endl;}

}


// Filter event
bool LightChHiggsToTauNuSkim::filter(Event& event, const EventSetup& setup ) {

  nEvents_++;
  int  nJets    = 0;
  int  nLeptons = 0;
  bool keepEvent = false;

  Handle<MuonCollection>  muons;
  event.getByLabel(muonsTag_, muons); 
   

  Handle<ElectronCollection> electrons;	
  event.getByLabel(electronsTag_,electrons);

  Handle< vector< Jet > > jets;
  event.getByLabel(jetsTag_, jets);

 
  //Process Muons
  if(muons.isValid()){
    MuonCollection::const_iterator muonIt;
    // Loop over muon collection
    for ( muonIt = muons->begin(); muonIt != muons->end(); ++muonIt ) {
      if ( muonIt->pt() > leptonPtMin_ && muonIt->eta() > leptonEtaMin_ && muonIt->eta() < leptonEtaMax_ ) nLeptons++; 
    }
  }


  // Process Electrons
  if(electrons.isValid()){

    ElectronCollection::const_iterator electronIt;
    // Loop over electron collection
    for ( electronIt = electrons->begin(); electronIt != electrons->end(); ++electronIt ) {
      if ( electronIt->pt() > leptonPtMin_ && electronIt->eta() > leptonEtaMin_ && electronIt->eta() < leptonEtaMax_ ) nLeptons++; 
    }
  }


  // Process Jets
  if(jets.isValid()){
    vector<Jet>::const_iterator jetIt;
    // Loop over jet collection
    for ( jetIt = jets->begin(); jetIt != jets->end(); ++jetIt ) {
      if ( jetIt->pt() > jetPtMin_ && jetIt->eta() > jetEtaMin_ && jetIt->eta() < jetEtaMax_ ) nJets++; 
    }
  }


  // Make decision
  if ( nLeptons >= 1 && nJets >= minNumbOfjets_) keepEvent = true;
  if (keepEvent) nSelectedEvents_++;

  return keepEvent;
}

