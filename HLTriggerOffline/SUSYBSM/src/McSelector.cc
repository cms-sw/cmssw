/*  \class McSelector
 *
 *  Class to apply analysis cuts in the TriggerValidation Code
 *
 *  Author: Massimiliano Chiorboli      Date: August 2007
 //         Maurizio Pierini
 //         Maria Spiropulu
 //    Philip Hebda, July 2009
*
 */

#include "HLTriggerOffline/SUSYBSM/interface/McSelector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace reco;
using namespace std;

McSelector::McSelector(edm::ParameterSet userCut_params)
{
  //******************** PLEASE PAY ATTENTION: number of electron and muons is strictly equal, for jets, taus and photons the requirement is >= ********************
  name     = userCut_params.getParameter<string>("name");
  m_genSrc       = userCut_params.getParameter<string>("mcparticles");
  m_genJetSrc    = userCut_params.getParameter<string>("genJets");
  m_genMetSrc    = userCut_params.getParameter<string>("genMet");
  mc_ptElecMin 	   = userCut_params.getParameter<double>("mc_ptElecMin"	    );
  mc_ptMuonMin 	   = userCut_params.getParameter<double>("mc_ptMuonMin"	    );
  mc_ptTauMin  	   = userCut_params.getParameter<double>("mc_ptTauMin" 	    );
  mc_ptPhotMin 	   = userCut_params.getParameter<double>("mc_ptPhotMin"	    );
  mc_ptJetMin  	   = userCut_params.getParameter<double>("mc_ptJetMin" 	    );   
  mc_ptJetForHtMin = userCut_params.getParameter<double>("mc_ptJetForHtMin" );
  mc_metMin        = userCut_params.getParameter<double>("mc_metMin"        );
  mc_htMin         = userCut_params.getParameter<double>("mc_htMin"         );
  mc_nElec     = userCut_params.getParameter<int>("mc_nElec");
  mc_nElecRule = userCut_params.getParameter<string>("mc_nElecRule");
  mc_nMuon     = userCut_params.getParameter<int>("mc_nMuon");
  mc_nMuonRule = userCut_params.getParameter<string>("mc_nMuonRule");
  mc_nTau      = userCut_params.getParameter<int>("mc_nTau");
  mc_nPhot     = userCut_params.getParameter<int>("mc_nPhot");
  mc_nJet      = userCut_params.getParameter<int>("mc_nJet" );


  edm::LogInfo("HLTriggerOfflineSUSYBSM") << "UserAnalysis parameters, MC for " << name << " selection:" ;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_ptElecMin      = " << mc_ptElecMin  	;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_ptMuonMin      = " << mc_ptMuonMin  	;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_ptTauMin       = " << mc_ptTauMin   	;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_ptPhotMin      = " << mc_ptPhotMin  	;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_ptJetMin       = " << mc_ptJetMin   	;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_ptJetForHtMin  = " << mc_ptJetForHtMin   ;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_metMin         = " << mc_metMin     	;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_htMin          = " << mc_htMin      	;

  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_nElec  	 = " << mc_nElec   ;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_nElecRule = " << mc_nElecRule ;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_nMuon  	 = " << mc_nMuon   ;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_nMuonRule = " << mc_nMuonRule ;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_nTau  	 = " << mc_nTau    ;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_nPhot  	 = " << mc_nPhot   ;
  edm::LogInfo("HLTriggerOfflineSUSYBSM") << " mc_nJet   	 = " << mc_nJet    ;


}

string McSelector::GetName(){return name;}

bool McSelector::isSelected(const edm::Event& iEvent)
{


  this->handleObjects(iEvent);


  bool TotalCutPassed = false;


  bool ElectronCutPassed = false;
  bool MuonCutPassed = false;
  bool TauCutPassed = false;
  bool PhotonCutPassed = false;
  bool JetCutPassed = false;  
  bool MetCutPassed = false;  
  bool HtCutPassed = false;  

  int nMcElectrons = 0;
  int nMcMuons     = 0;
  int nMcTaus     = 0;
  int nMcPhotons   = 0;
  int nMcJets      = 0;
  
  //  cout <<"----------------------------------------------------------------------------" << endl;
  
  for(unsigned int i=0; i<theGenParticleCollection->size(); i++) {
    const GenParticle* genParticle = (&(*theGenParticleCollection)[i]);    
    if(genParticle->status() == 2) {
      //taus
      if(fabs(genParticle->pdgId()) == 15) {
	//	cout << "Tau Mother = " << genParticle->mother()->pdgId() << endl;
	//	if(fabs(genParticle->mother()->pdgId()) == 15) 	cout << "Tau GrandMother = " << genParticle->mother()->mother()->pdgId() << endl;
	if(genParticle->pt() > mc_ptTauMin && fabs(genParticle->eta())<2.5) {
	  nMcTaus++;
	}
      }
    }
    if(genParticle->status() == 1) {
      //      cout << "genParticle->status() = " << genParticle->status() << "    genParticle->pdgId() = " << genParticle->pdgId() << "     genParticle->pt() = " << genParticle->pt() << endl;
      //electrons
      if(fabs(genParticle->pdgId()) == 11 && genParticle->numberOfMothers()) {
	//		cout << "Mc Electron, pt = " << genParticle->pt() << endl;
	//		cout << "Electron Mother = " << genParticle->mother()->pdgId() << endl;
// 		if(fabs(genParticle->mother()->pdgId()) == 11 || fabs(genParticle->mother()->pdgId()) == 15) 
// 		  cout << "Electron GrandMother = " << genParticle->mother()->mother()->pdgId() << endl;
// 		if(fabs(genParticle->mother()->mother()->pdgId()) == 11 || fabs(genParticle->mother()->mother()->pdgId()) == 15)
// 		  cout << "Electron GrandGrandMother = " << genParticle->mother()->mother()->mother()->pdgId() << endl;
	int motherCode = genParticle->mother()->pdgId();
	if(fabs(motherCode) == fabs(genParticle->pdgId()) || fabs(motherCode) == 15) {
	  motherCode = genParticle->mother()->mother()->pdgId();
	  if(fabs(motherCode) == fabs(genParticle->pdgId()) || fabs(motherCode) == 15) motherCode = genParticle->mother()->mother()->mother()->pdgId();
	}
	if(fabs(motherCode) == 23 || fabs(motherCode) == 24 || fabs(motherCode) > 999999) {
	  if(genParticle->pt() > mc_ptElecMin && fabs(genParticle->eta())<2.5) {
	    nMcElectrons++;
	    //	  	  cout << "Mc Electron, pt = " << genParticle->pt() << endl;
	  }
	}
      }
      //muons
      if(fabs(genParticle->pdgId()) == 13 && genParticle->numberOfMothers()) {
	//	cout << "Mc Muon, pt = " << genParticle->pt() << endl;
	//		cout << "Muon Mother = " << genParticle->mother()->pdgId() << endl;
// 		if(fabs(genParticle->mother()->pdgId()) == 13 || fabs(genParticle->mother()->pdgId()) == 15)
// 		  cout << "Muon GrandMother = " << genParticle->mother()->mother()->pdgId() << endl;
// 		if(fabs(genParticle->mother()->mother()->pdgId()) == 13 || fabs(genParticle->mother()->mother()->pdgId()) == 15)
// 		  cout << "Muon GrandGrandMother = " << genParticle->mother()->mother()->mother()->pdgId() << endl;
	int motherCode = genParticle->mother()->pdgId();
	if(fabs(motherCode) == fabs(genParticle->pdgId()) || fabs(motherCode) == 15) {
	  motherCode = genParticle->mother()->mother()->pdgId();
	  if(fabs(motherCode) == fabs(genParticle->pdgId()) || fabs(motherCode) == 15) motherCode = genParticle->mother()->mother()->mother()->pdgId();
	}
	if(fabs(motherCode) == 23 || fabs(motherCode) == 24 || fabs(motherCode) > 999999) {
	  if(genParticle->pt() > mc_ptMuonMin && fabs(genParticle->eta())<2.5) {
	    nMcMuons++;
	  //	  	  cout << "Mc Muon, pt = " << genParticle->pt() << endl;
	  }
	}
      }
      //photons
      if(fabs(genParticle->pdgId()) == 22 && fabs(genParticle->eta())<2.5) {
	//	cout << "Mc Photon, pt = " << genParticle->pt() << endl;
	if(genParticle->pt() > mc_ptPhotMin) {
	  //	  cout << "Photon Mother = " << genParticle->mother()->pdgId() << endl;
	  nMcPhotons++;
	  //	  cout << "Mc Photon, pt = " << genParticle->pt() << endl;
	}
      }
    }
  }

  ht = 0;
  for(unsigned int i=0; i<theGenJetCollection->size();i++) {
    GenJet genjet = (*theGenJetCollection)[i];
    //    cout << "Mc Jet, pt = " << genjet.pt() << endl;
    if(genjet.pt() > mc_ptJetMin && fabs(genjet.eta())<3) {
      nMcJets++;
      //           cout << "Mc Jet, pt = " << genjet.pt() << endl;
    }
    if(genjet.pt() > mc_ptJetForHtMin) ht =+ genjet.pt();
  }
  if(ht>mc_htMin) HtCutPassed = true;
  
  
  const GenMET theGenMET = theGenMETCollection->front();
  //  cout << "GenMET = " << theGenMET.pt() << endl;
  if(theGenMET.pt() > mc_metMin) MetCutPassed = true;
  
  //  cout <<"----------------------------------------------------------------------------" << endl;
  

//   cout << "nMcElectrons = " << nMcElectrons << endl;
//   cout << "nMcMuons     = " << nMcMuons     << endl;
//   cout << "nMcTaus      = " << nMcTaus      << endl;
//   cout << "nMcPhotons   = " << nMcPhotons   << endl;
//   cout << "nMcJets      = " << nMcJets      << endl;
  
  if(mc_nElecRule == "strictEqual") {
    if(nMcElectrons == mc_nElec) ElectronCutPassed = true;
  } 
  else if (mc_nElecRule == "greaterEqual") {
    if(nMcElectrons >= mc_nElec) ElectronCutPassed = true;
  } 
  else cout << "WARNING: not a correct definition of cuts at gen level for electrons! " << endl;
  
  
  if(mc_nMuonRule == "strictEqual") {
    if(nMcMuons     == mc_nMuon) MuonCutPassed     = true;
  }
  else if (mc_nMuonRule == "greaterEqual") {
    if(nMcMuons >= mc_nMuon) MuonCutPassed = true;
  } 
  else cout << "WARNING: not a correct definition of cuts at gen level for muons! " << endl;
  if(nMcTaus      >= mc_nTau)  TauCutPassed      = true;
  if(nMcPhotons   >= mc_nPhot) PhotonCutPassed   = true;
  if(nMcJets      >= mc_nJet ) JetCutPassed      = true;
  
  
//   cout << "ElectronCutPassed = " << (int)ElectronCutPassed << endl;
//   cout << "MuonCutPassed     = " << (int)MuonCutPassed << endl;
//   cout << "PhotonCutPassed   = " << (int)PhotonCutPassed << endl;
//   cout << "JetCutPassed      = " << (int)JetCutPassed << endl;
//   cout << "MetCutPassed      = " << (int)MetCutPassed << endl;
  
  
//   if(
//      (ElectronCutPassed || MuonCutPassed) &&
//      PhotonCutPassed                      &&
//      JetCutPassed                         &&
//      MetCutPassed      )   TotalCutPassed = true;
  
  
  if(
     ElectronCutPassed   && 
     MuonCutPassed       &&
     TauCutPassed        &&
     PhotonCutPassed     &&
     JetCutPassed        &&
     MetCutPassed        &&
     HtCutPassed          )   TotalCutPassed = true;


  // Apply a veto: removed because we require the exact number of leptons
  // and not >=
  // the veto is hence equivalent to request 0 leptons

//   if(TotalCutPassed) {
//     if(mc_ElecVeto) {
//       if(nMcElectrons>0) TotalCutPassed = false;
//     }
//     if(mc_MuonVeto) {
//       if(nMcMuons>0) TotalCutPassed = false;
//     }
//   }

  
//  cout << "TotalCutPassed = " << TotalCutPassed << endl;

  return TotalCutPassed;
}
 
void McSelector::handleObjects(const edm::Event& iEvent)
{

  //Get the GenParticleCandidates
  Handle< reco::GenParticleCollection > theCandidateCollectionHandle;
  iEvent.getByLabel(m_genSrc, theCandidateCollectionHandle);
  theGenParticleCollection = theCandidateCollectionHandle.product();

  
  //Get the GenJets
  Handle< GenJetCollection > theGenJetCollectionHandle ;
  iEvent.getByLabel( m_genJetSrc, theGenJetCollectionHandle);  
  theGenJetCollection = theGenJetCollectionHandle.product();


  //Get the GenMET
  Handle< GenMETCollection > theGenMETCollectionHandle;
  iEvent.getByLabel( m_genMetSrc, theGenMETCollectionHandle);
  theGenMETCollection = theGenMETCollectionHandle.product();


}
