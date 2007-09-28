/*  \class CutSelector
*
*  Class to apply analysis cuts in the TriggerValidation Code
*
*  Author: Massimiliano Chiorboli      Date: August 2007
*
*/

#include "HLTriggerOffline/SUSYBSM/interface/CutSelector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace reco;
using namespace std;

CutSelector::CutSelector(edm::ParameterSet userCut_params)
{

  m_electronSrc  = userCut_params.getParameter<string>("electrons");
  m_muonSrc    	 = userCut_params.getParameter<string>("muons");
  m_jetsSrc    	 = userCut_params.getParameter<string>("jets");
  m_photonSrc  	 = userCut_params.getParameter<string>("photons");
  m_calometSrc 	 = userCut_params.getParameter<string>("calomet");

  user_metMin    = userCut_params.getParameter<double>("user_metMin") ;
  user_ptJet1Min = userCut_params.getParameter<double>("user_ptJet1Min");
  user_ptJet2Min = userCut_params.getParameter<double>("user_ptJet2Min");
  user_ptElecMin = userCut_params.getParameter<double>("user_ptElecMin");
  user_ptMuonMin = userCut_params.getParameter<double>("user_ptMuonMin");
  user_ptPhotMin = userCut_params.getParameter<double>("user_ptPhotMin");

  cout << endl;
  cout << "UserAnalysis parameters:" << endl;
  cout << " user_metMin    = " << user_metMin << endl;
  cout << " user_ptJet1Min = " << user_ptJet1Min  << endl;
  cout << " user_ptJet2Min = " << user_ptJet2Min  << endl;
  cout << " user_ptElecMin = " << user_ptElecMin  << endl;
  cout << " user_ptMuonMin = " << user_ptMuonMin  << endl;
  cout << " user_ptPhotMin = " << user_ptPhotMin  << endl;

}

bool CutSelector::isSelected(const edm::Event& iEvent)
{



  this->handleObjects(iEvent);



  bool TotalCutPassed = false;

  bool ElectronCutPassed = false;
  for(unsigned int i=0; i<theElectronCollection->size(); i++) {
    PixelMatchGsfElectron electron = (*theElectronCollection)[i];
    float elenergy = electron.superCluster()->energy();
    float elpt = electron.pt() * elenergy / electron.p();
    if(elpt>user_ptElecMin)  ElectronCutPassed = true;
    //    cout << "elpt = " << elpt << endl;
  }

  
  bool MuonCutPassed = false;
  for(unsigned int i=0; i<theMuonCollection->size(); i++) {
    Muon muon = (*theMuonCollection)[i];
    float muonpt = muon.pt();
    if(muonpt>user_ptMuonMin) MuonCutPassed = true;
    //    cout << "muonpt = " << muonpt << endl;
  }
  
  bool PhotonCutPassed = false;
  for(unsigned int i=0; i<thePhotonCollection->size(); i++) {
    Photon photon = (*thePhotonCollection)[i];
    float photonpt = photon.pt();
    if(photonpt>user_ptPhotMin) PhotonCutPassed = true;
    //    cout << "photonpt = " << photonpt << endl;
  }


  bool JetCutPassed = false;  
  for(unsigned int i=0; i<theCaloJetCollection->size(); i++) {
    Jet jet = (*theCaloJetCollection)[i];
    float jetpt = jet.pt();
    //    cout << "jetpt = " << jetpt << endl;
  }
  if(theCaloJetCollection->size()>1) {
    if((*theCaloJetCollection)[0].pt() > user_ptJet1Min &&
       (*theCaloJetCollection)[1].pt() > user_ptJet2Min ) JetCutPassed = true;
  }

  bool MetCutPassed = false;  
  const CaloMET theCaloMET = theCaloMETCollection->front();
  float calomet = theCaloMET.pt();
  //  cout << "calomet = " << calomet << endl;
  if(calomet > user_metMin) MetCutPassed = true;

//   cout << "ElectronCutPassed = " << (int)ElectronCutPassed << endl;
//   cout << "MuonCutPassed     = " << (int)MuonCutPassed << endl;
//   cout << "PhotonCutPassed   = " << (int)PhotonCutPassed << endl;
//   cout << "JetCutPassed      = " << (int)JetCutPassed << endl;
//   cout << "MetCutPassed      = " << (int)MetCutPassed << endl;



  if(
     (ElectronCutPassed || MuonCutPassed) &&
     PhotonCutPassed                      &&
     JetCutPassed                         &&
     MetCutPassed      )   TotalCutPassed = true;


  return TotalCutPassed;
}
 
void CutSelector::handleObjects(const edm::Event& iEvent)
{

  //Get the electrons
  Handle<PixelMatchGsfElectronCollection> theElectronCollectionHandle; 
  iEvent.getByLabel(m_electronSrc, theElectronCollectionHandle);
  theElectronCollection = theElectronCollectionHandle.product();

  //Get the Muons
  Handle<MuonCollection> theMuonCollectionHandle; 
  iEvent.getByLabel(m_muonSrc, theMuonCollectionHandle);
  theMuonCollection = theMuonCollectionHandle.product();

  //Get the Photons
  Handle<PhotonCollection> thePhotonCollectionHandle; 
  iEvent.getByLabel(m_photonSrc, thePhotonCollectionHandle);
  thePhotonCollection = thePhotonCollectionHandle.product();

  //Get the CaloJets
  Handle<CaloJetCollection> theCaloJetCollectionHandle;
  iEvent.getByLabel(m_jetsSrc, theCaloJetCollectionHandle);
  theCaloJetCollection = theCaloJetCollectionHandle.product();

  //Get the CaloMET
  Handle<CaloMETCollection> theCaloMETCollectionHandle;
  iEvent.getByLabel(m_calometSrc, theCaloMETCollectionHandle);
  theCaloMETCollection = theCaloMETCollectionHandle.product();

}
