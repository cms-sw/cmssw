/*  \class RecoSelector
*
*  Class to apply analysis cuts in the TriggerValidation Code
*
*  Author: Massimiliano Chiorboli      Date: August 2007
//         Maurizio Pierini
//         Maria Spiropulu
*
*/

#include "HLTriggerOffline/SUSYBSM/interface/RecoSelector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace reco;
using namespace std;

RecoSelector::RecoSelector(edm::ParameterSet userCut_params)
{

  name                   = userCut_params.getParameter<string>("name");
  m_electronSrc  	 = userCut_params.getParameter<string>("electrons");
  m_muonSrc    	 	 = userCut_params.getParameter<string>("muons");
  m_jetsSrc    	 	 = userCut_params.getParameter<string>("jets");
  m_photonProducerSrc  	 = userCut_params.getParameter<string>("photonProducer");
  m_photonSrc  	 	 = userCut_params.getParameter<string>("photons");
  m_calometSrc 	         = userCut_params.getParameter<string>("calomet");

  reco_metMin    = userCut_params.getParameter<double>("reco_metMin") ;
  reco_ptJet1Min = userCut_params.getParameter<double>("reco_ptJet1Min");
  reco_ptJet2Min = userCut_params.getParameter<double>("reco_ptJet2Min");
  reco_ptElecMin = userCut_params.getParameter<double>("reco_ptElecMin");
  reco_ptMuonMin = userCut_params.getParameter<double>("reco_ptMuonMin");
  reco_ptPhotMin = userCut_params.getParameter<double>("reco_ptPhotMin");

  edm::LogInfo("RecoSelectorParameters") << endl;
  edm::LogInfo("RecoSelectorParameters") << "UserAnalysis parameters for " << name << " selection:" << endl;
  edm::LogInfo("RecoSelectorParameters") << " reco_metMin    = " << reco_metMin << endl;
  edm::LogInfo("RecoSelectorParameters") << " reco_ptJet1Min = " << reco_ptJet1Min  << endl;
  edm::LogInfo("RecoSelectorParameters") << " reco_ptJet2Min = " << reco_ptJet2Min  << endl;
  edm::LogInfo("RecoSelectorParameters") << " reco_ptElecMin = " << reco_ptElecMin  << endl;
  edm::LogInfo("RecoSelectorParameters") << " reco_ptMuonMin = " << reco_ptMuonMin  << endl;
  edm::LogInfo("RecoSelectorParameters") << " reco_ptPhotMin = " << reco_ptPhotMin  << endl;

}

string RecoSelector::GetName(){return name;}

bool RecoSelector::isSelected(const edm::Event& iEvent)
{


  this->handleObjects(iEvent);


  bool TotalCutPassed = false;

  bool ElectronCutPassed = false;
  for(unsigned int i=0; i<theElectronCollection->size(); i++) {
    GsfElectron electron = (*theElectronCollection)[i];
    if (electron.ecalDrivenSeed()) {
      float elenergy = electron.superCluster()->energy();
      float elpt = electron.pt() * elenergy / electron.p();
      if(elpt>reco_ptElecMin)  ElectronCutPassed = true;
      LogDebug("RecoSelectorCuts") << "elpt = " << elpt << endl;
    }
    else {
      //      float elenergy = 0;
      //      float elpt = 0;
      ElectronCutPassed = false;
    }
  }

  bool MuonCutPassed = false;
  for(unsigned int i=0; i<theMuonCollection->size(); i++) {
    Muon muon = (*theMuonCollection)[i];
    float muonpt = muon.pt();
    if(muonpt>reco_ptMuonMin) MuonCutPassed = true;
    LogDebug("RecoSelectorCuts") << "muonpt = " << muonpt << endl;
  }
  
  bool PhotonCutPassed = false;
  for(unsigned int i=0; i<thePhotonCollection->size(); i++) {
    Photon photon = (*thePhotonCollection)[i];
    float photonpt = photon.pt();
    if(photonpt>reco_ptPhotMin) PhotonCutPassed = true;
    LogDebug("RecoSelectorCuts") << "photonpt = " << photonpt << endl;
  }


  bool JetCutPassed = false;  
  if(theCaloJetCollection->size()>1) {
    if((*theCaloJetCollection)[0].pt() > reco_ptJet1Min &&
       (*theCaloJetCollection)[1].pt() > reco_ptJet2Min ) JetCutPassed = true;
  }

  bool MetCutPassed = false;  
  const CaloMET theCaloMET = theCaloMETCollection->front();
  float calomet = theCaloMET.pt();
  LogDebug("RecoSelectorCuts") << "calomet = " << calomet << endl;
  if(calomet > reco_metMin) MetCutPassed = true;

  edm::LogInfo("RecoSelectorPassed") << "ElectronCutPassed = " << (int)ElectronCutPassed << endl;
  edm::LogInfo("RecoSelectorPassed") << "MuonCutPassed     = " << (int)MuonCutPassed << endl;
  edm::LogInfo("RecoSelectorPassed") << "PhotonCutPassed   = " << (int)PhotonCutPassed << endl;
  edm::LogInfo("RecoSelectorPassed") << "JetCutPassed      = " << (int)JetCutPassed << endl;
  edm::LogInfo("RecoSelectorPassed") << "MetCutPassed      = " << (int)MetCutPassed << endl;


  if(
     (ElectronCutPassed || MuonCutPassed) &&
     PhotonCutPassed                      &&
     JetCutPassed                         &&
     MetCutPassed      )   TotalCutPassed = true;

  return TotalCutPassed;
}
 
void RecoSelector::handleObjects(const edm::Event& iEvent)
{

  //Get the electrons
  Handle<GsfElectronCollection> theElectronCollectionHandle; 
  iEvent.getByLabel(m_electronSrc, theElectronCollectionHandle);
  theElectronCollection = theElectronCollectionHandle.product();

  //Get the Muons
  Handle<MuonCollection> theMuonCollectionHandle; 
  iEvent.getByLabel(m_muonSrc, theMuonCollectionHandle);
  theMuonCollection = theMuonCollectionHandle.product();

  //Get the Photons
  Handle<PhotonCollection> thePhotonCollectionHandle; 
  iEvent.getByLabel(m_photonProducerSrc, m_photonSrc, thePhotonCollectionHandle);
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
