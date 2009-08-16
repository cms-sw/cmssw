#ifndef LightChHiggsToTauNuSkim_h
#define LightChHiggsToTauNuSkim_h

/** \class HeavyChHiggsToTauNuSkim
 *
 *  
 *  Filter to select events passing 
 *  HLT muon/electron trigger
 *  with at least one offline lepton and two jets
 *
 *  \author Nuno Almeida  -  LIP Lisbon
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


// Reco Objects ///////////////////////////////////////////
// Muons
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

// Electrons
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
//#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
//#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"

#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDFwd.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"
// Jets
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
///////////////////////////////////////////////////////////


#include <TLorentzVector.h>


// trigger ////////////////////////////////////////
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
////////////////////////////////////////////////////



using namespace edm;
using namespace std;
using namespace reco;
using namespace trigger;


class LightChHiggsToTauNuSkim : public edm::EDFilter {

    public:
     explicit LightChHiggsToTauNuSkim(const edm::ParameterSet&);
     ~LightChHiggsToTauNuSkim();

     virtual bool filter(edm::Event&, const edm::EventSetup& );

   private:

     InputTag jetsTag_;
     InputTag muonsTag_;
     InputTag electronsTag_;
     InputTag electronId_;
     InputTag triggersEventTag_;
    
     int minNumbOfjets_;
     double jetPtMin_;
     double jetEtaMin_;
     double jetEtaMax_;

     double leptonPtMin_;
     double leptonEtaMin_;
     double leptonEtaMax_;

     double drHLT_;
     double drHLTMatch_;
   
     vector<string> hltFiltersByName_;

     double nEvents_;
     double nSelectedEvents_;

     bool applyElectronId_;
};
#endif
