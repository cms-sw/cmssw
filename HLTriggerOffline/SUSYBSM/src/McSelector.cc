/*  \class McSelector
 *
 *  Class to apply analysis cuts in the TriggerValidation Code
 *
 *  Author: Massimiliano Chiorboli      Date: August 2007
 //         Maurizio Pierini
 //         Maria Spiropulu
*
 */

#include "HLTriggerOffline/SUSYBSM/interface/McSelector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace reco;
using namespace std;

McSelector::McSelector(edm::ParameterSet userCut_params)
{

  m_genSrc       = userCut_params.getParameter<string>("mcparticles");
  m_genJetSrc    = userCut_params.getParameter<string>("genJets");
  m_genMetSrc    = userCut_params.getParameter<string>("genMet");

  mc_ptElecMin = userCut_params.getParameter<double>("mc_ptElecMin");
  mc_ptMuonMin = userCut_params.getParameter<double>("mc_ptMuonMin");
  mc_ptPhotMin = userCut_params.getParameter<double>("mc_ptPhotMin");
  mc_ptJetMin  = userCut_params.getParameter<double>("mc_ptJetMin" );
  mc_metMin    = userCut_params.getParameter<double>("mc_metMin"   );

  mc_nElec     = userCut_params.getParameter<int>("mc_nElec");
  mc_nMuon     = userCut_params.getParameter<int>("mc_nMuon");
  mc_nPhot     = userCut_params.getParameter<int>("mc_nPhot");
  mc_nJet      = userCut_params.getParameter<int>("mc_nJet" );



  cout << endl;
  cout << "UserAnalysis parameters, MC:" << endl;
  cout << " mc_ptElecMin = " << mc_ptElecMin  << endl;
  cout << " mc_ptMuonMin = " << mc_ptMuonMin  << endl;
  cout << " mc_ptPhotMin = " << mc_ptPhotMin  << endl;
  cout << " mc_ptJetMin  = " << mc_ptJetMin   << endl;
  cout << " mc_metMin    = " << mc_metMin     << endl;

  cout << " mc_nElec  	 = " << mc_nElec   << endl;
  cout << " mc_nMuon  	 = " << mc_nMuon   << endl;
  cout << " mc_nPhot  	 = " << mc_nPhot   << endl;
  cout << " mc_nJet   	 = " << mc_nJet    << endl;
  cout << endl;

}

bool McSelector::isSelected(const edm::Event& iEvent)
{


  this->handleObjects(iEvent);


  bool TotalCutPassed = false;


  bool ElectronCutPassed = false;
  bool MuonCutPassed = false;
  bool PhotonCutPassed = false;
  bool JetCutPassed = false;  
  bool MetCutPassed = false;  

  int nMcElectrons = 0;
  int nMcMuons     = 0;
  int nMcPhotons   = 0;
  int nMcJets      = 0;
  for(unsigned int i=0; i<theGenParticleCollection->size(); i++) {
    //    const GenParticleCandidate* genParticle = dynamic_cast<const GenParticleCandidate*> (&(*theGenParticleCollection)[i]);    
    const GenParticle* genParticle = (&(*theGenParticleCollection)[i]);    
    if(genParticle->status() == 1) {
      //electrons
      if(fabs(genParticle->pdgId()) == 11) {
	if(genParticle->pt() > mc_ptElecMin) {
	  nMcElectrons++;
	  //	  cout << "Mc Electron, pt = " << genParticle->pt() << endl;
	}
      }
      //muons
      if(fabs(genParticle->pdgId()) == 13) {
	if(genParticle->pt() > mc_ptMuonMin) {
	  nMcMuons++;
	  //	  cout << "Mc Muon, pt = " << genParticle->pt() << endl;
	}
      }
      //photons
      if(fabs(genParticle->pdgId()) == 22) {
	if(genParticle->pt() > mc_ptPhotMin) {
	  nMcPhotons++;
	  //	  cout << "Mc Photon, pt = " << genParticle->pt() << endl;      }
	}
      }
    }
  }

  for(unsigned int i=0; i<theGenJetCollection->size();i++) {
    GenJet genjet = (*theGenJetCollection)[i];
    if(genjet.pt() > mc_ptJetMin) {
      nMcJets++;
      //      cout << "Mc Jet, pt = " << genjet.pt() << endl;
    }
  }


  const GenMET theGenMET = theGenMETCollection->front();
  //  cout << "GenMET = " << theGenMET.pt() << endl;
  if(theGenMET.pt() > mc_metMin) MetCutPassed = true;


  
  if(nMcElectrons >= mc_nElec) ElectronCutPassed = true;
  if(nMcMuons     >= mc_nMuon) MuonCutPassed     = true;
  if(nMcPhotons   >= mc_nPhot) PhotonCutPassed   = true;
  if(nMcJets      >= mc_nJet ) JetCutPassed      = true;


  cout << "ElectronCutPassed = " << (int)ElectronCutPassed << endl;
  cout << "MuonCutPassed     = " << (int)MuonCutPassed << endl;
  cout << "PhotonCutPassed   = " << (int)PhotonCutPassed << endl;
  cout << "JetCutPassed      = " << (int)JetCutPassed << endl;
  cout << "MetCutPassed      = " << (int)MetCutPassed << endl;


  if(
     (ElectronCutPassed || MuonCutPassed) &&
     PhotonCutPassed                      &&
     JetCutPassed                         &&
     MetCutPassed      )   TotalCutPassed = true;



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
