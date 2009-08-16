
/** \class LightChHiggsToTauNuSkim
 *
 * Consult header file for description
 *
 * \author:  Nuno Almeida - LIP-Lisbon
 */
 



#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HiggsAnalysis/Skimming/interface/LightChHiggsToTauNuSkim.h"

/*

// Muons
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"


// Electrons
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

// Jets
#include "DataFormats/JetReco/interface/Jet.h"
*/

#include "DataFormats/Math/interface/deltaR.h"

using namespace std;
using namespace edm;
using namespace reco;




// Constructor
LightChHiggsToTauNuSkim::LightChHiggsToTauNuSkim(const edm::ParameterSet& cfg) :

  jetsTag_         (cfg.getParameter<InputTag>("jetsTag")),
  muonsTag_        (cfg.getParameter<InputTag>("muonsTag")),
  electronsTag_    (cfg.getParameter<InputTag>("electronsTag")),
  electronId_      (cfg.getParameter<InputTag>("electronIdTag")),
  triggersEventTag_(cfg.getParameter<InputTag>("triggerEventTag")),

  minNumbOfjets_   (cfg.getParameter<int>("minNumbOfJets")),
 
  jetPtMin_        (cfg.getParameter<double>("jetPtMin")),
  jetEtaMin_       (cfg.getParameter<double>("jetEtaMin")),
  jetEtaMax_       (cfg.getParameter<double>("jetEtaMax")),

  
  leptonPtMin_     (cfg.getParameter<double>("leptonPtMin")),
  leptonEtaMin_    (cfg.getParameter<double>("leptonEtaMin")),
  leptonEtaMax_    (cfg.getParameter<double>("leptonEtaMax")),


  drHLT_           (cfg.getParameter<double>("drHLT")),
  drHLTMatch_      (cfg.getParameter<double>("drHLTMatch")),


  hltFiltersByName_ (cfg.getParameter< vector<string > > ("hltFilters"))

{
  nEvents_         = 0;
  nSelectedEvents_ = 0;
  if(electronId_ == InputTag("none")){ applyElectronId_ = false; }
  else{ applyElectronId_ = true;}
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
  int nJets = 0;

  Handle<MuonCollection>  muons;
  event.getByLabel(muonsTag_, muons); 

  Handle<GsfElectronCollection> electrons;	
  event.getByLabel(electronsTag_,electrons);

  Handle< CaloJetCollection > jets;
  event.getByLabel(jetsTag_, jets);


  //FIND HLT Filter Objects
  edm::Handle<trigger::TriggerEvent> TriggerEventHandle;
  event.getByLabel(triggersEventTag_,TriggerEventHandle);

  TLorentzVector P_TriggerLeptons_temp;
  vector<TLorentzVector> P_TriggerLeptons;

  if (TriggerEventHandle.isValid()) {

    const size_type nO(TriggerEventHandle->sizeObjects());
    const TriggerObjectCollection& TOC(TriggerEventHandle->getObjects());
    for (size_type iO=0; iO!=nO; ++iO) {}
    const size_type nF(TriggerEventHandle->sizeFilters());

    for (size_type iF=0; iF!=nF; ++iF) {
      edm::InputTag triggerlabelname = TriggerEventHandle->filterTag(iF);

      for (std::vector<std::string>::const_iterator iL =  hltFiltersByName_.begin(); iL !=  hltFiltersByName_.end(); ++iL) {

        //LOOP over right filter names
        std::string filterString = (*iL);

        //cout<<endl<<" triggerlabelname.label()  "<<(triggerlabelname.label())<<" filterString "<<filterString<<endl;

        if( triggerlabelname.label() == filterString ) {

          // cout<<endl<<" filter trigger : "<<filterString<<endl;

          //cout<<endl<<" triggered label : "<<filterString<<endl;
          const Keys& KEYS(TriggerEventHandle->filterKeys(iF));
          const Vids& VIDS (TriggerEventHandle->filterIds(iF));
          const size_type nI(VIDS.size());
          const size_type nK(KEYS.size());
          const size_type n(std::max(nI,nK));

          for (size_type i=0; i!=n; ++i) {
            P_TriggerLeptons_temp.SetPxPyPzE(TOC[KEYS[i]].px(),TOC[KEYS[i]].py(),TOC[KEYS[i]].pz(),TOC[KEYS[i]].energy());
            P_TriggerLeptons.push_back(P_TriggerLeptons_temp);
          }
        }
      }
    }
  }

 
  //FIND the highest pt trigger lepton //////////////////////////////////////////////////////////////////////////////////////////////////////
  TLorentzVector theTriggerLepton;
  double maxPtTrigger = 0;

  //cout<<endl<<" Number of lepton triggers : "<<(P_TriggerLeptons.size())<<endl;
  for(vector<TLorentzVector>::const_iterator triggerLeptons_iter = P_TriggerLeptons.begin();
    triggerLeptons_iter != P_TriggerLeptons.end(); triggerLeptons_iter++){
    if( triggerLeptons_iter->Perp() >= maxPtTrigger) {
      maxPtTrigger = triggerLeptons_iter->Perp();
      theTriggerLepton.SetPxPyPzE(triggerLeptons_iter->Px(), triggerLeptons_iter->Py(),triggerLeptons_iter->Pz(), triggerLeptons_iter->E());
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 
  // We require highest pt of triggered lepton to be > leptonPtMin 
  if( maxPtTrigger < leptonPtMin_){return false;}
  ////////////////////////////////////////////////////////////////

  
  double triggerEta = theTriggerLepton.Eta();
  double triggerPhi = theTriggerLepton.Phi();
     

   // Process Jets ////////////////////////////////////////////////////////////////////////////////
   if(jets.isValid()){
    vector<CaloJet>::const_iterator jetIt;
    // Loop over jet collection
    for ( jetIt = jets->begin(); jetIt != jets->end(); ++jetIt ) {
      if ( jetIt->pt() > jetPtMin_ && jetIt->eta() > jetEtaMin_ && jetIt->eta() < jetEtaMax_ ) {
        double dr = deltaR(jetIt->eta(),jetIt->phi(), triggerEta, triggerPhi); 
        if(dr>drHLT_) nJets++;
      }
    }
  }
  if ( nJets <minNumbOfjets_){return false;} 
  ///////////////////////////////////////////////////////////////////////////////////////////////////


  //Find the lepton that is closest matched with the triggered lepton
  double drMin = drHLTMatch_;
  // Loop over electrons
  vector<GsfElectron>::const_iterator eIt;
  int i=0;
  for( eIt = electrons->begin(); eIt != electrons->end(); eIt++, i++){
   
    // applyElectronId if needed;
    if(applyElectronId_){  
      Ref<GsfElectronCollection> electronRef(electrons,i);
      vector<Handle<ValueMap<float> > > eIDValueMap(4);
      event.getByLabel( electronId_ , eIDValueMap[3] );  
      const ValueMap<float> & eIDmap = * eIDValueMap[3];
      if( ! eIDmap[electronRef] ) {continue;}
    }


    if( eIt->pt() > leptonPtMin_  ){
      double eEta = eIt->eta(); double ePhi = eIt->phi();
      double dr = deltaR(eEta,ePhi,triggerEta,triggerPhi);
      if( dr < drMin){  drMin = dr;}
    }
  }


  // Loop over muons
  vector<Muon>::const_iterator mIt;
  for( mIt = muons->begin(); mIt != muons->end(); mIt++){

    if( mIt->pt() > leptonPtMin_  ){
      double mEta = mIt->eta(); double mPhi = mIt->phi();
      double dr = deltaR(mEta,mPhi,triggerEta,triggerPhi);
      if( dr < drMin){  drMin = dr; }
    }
  }
  ////////////////////////////////////////////////////////////////////

  // if a match is not found we skip event
  if(drMin==drHLTMatch_){return false;}
  else{ nSelectedEvents_++; return true;} 
  //////////////////////////////////////// 


}

