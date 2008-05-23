/*  \class PlotMaker
*
*  Class to produce some plots of Off-line variables in the TriggerValidation Code
*
*  Author: Massimiliano Chiorboli      Date: September 2007
//         Maurizio Pierini
//         Maria Spiropulu
*
*/
#include "HLTriggerOffline/SUSYBSM/interface/PlotMaker.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TDirectory.h"

#include "HLTriggerOffline/SUSYBSM/interface/PtSorter.h"


using namespace edm;
using namespace reco;
using namespace std;
using namespace l1extra;

PlotMaker::PlotMaker(edm::ParameterSet objectList)
{
  m_l1extra      	 = objectList.getParameter<string>("l1extramc");
  m_electronSrc  	 = objectList.getParameter<string>("electrons");
  m_muonSrc    	 	 = objectList.getParameter<string>("muons");
  m_jetsSrc    	 	 = objectList.getParameter<string>("jets");
  m_photonProducerSrc  	 = objectList.getParameter<string>("photonProducer");
  m_photonSrc  	 	 = objectList.getParameter<string>("photons");
  m_calometSrc 	 	 = objectList.getParameter<string>("calomet");

  def_electronPtMin = objectList.getParameter<double>("def_electronPtMin");
  def_muonPtMin     = objectList.getParameter<double>("def_muonPtMin");
  def_jetPtMin      = objectList.getParameter<double>("def_jetPtMin");
  def_photonPtMin   = objectList.getParameter<double>("def_photonPtMin");

  cout << endl;
  cout << "Object definition cuts:" << endl;
  cout << " def_electronPtMin  " << def_electronPtMin << endl;
  cout << " def_muonPtMin      " << def_muonPtMin     << endl;
  cout << " def_jetPtMin       " << def_jetPtMin      << endl;
  cout << " def_photonPtMin    " << def_photonPtMin   << endl;


}

void PlotMaker::fillPlots(const edm::Event& iEvent)
{
  this->handleObjects(iEvent);



  //**********************
  // Fill the L1 Object Histos
  //**********************

  //**********************
  // Fill the Jet Histos
  //**********************

  

  hL1CentralJetMult->Fill(theL1CentralJetCollection.size());
  if(theL1CentralJetCollection.size()>0) {
    hL1CentralJet1Pt->Fill(theL1CentralJetCollection[0].pt());
    hL1CentralJet1Eta->Fill(theL1CentralJetCollection[0].eta());
    hL1CentralJet1Phi->Fill(theL1CentralJetCollection[0].phi());
  }
  if(theL1CentralJetCollection.size()>1) {
    hL1CentralJet2Pt->Fill(theL1CentralJetCollection[1].pt());
    hL1CentralJet2Eta->Fill(theL1CentralJetCollection[1].eta());
    hL1CentralJet2Phi->Fill(theL1CentralJetCollection[1].phi());
  }
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1CentralJetMultAfterL1[i]->Fill(theL1CentralJetCollection.size());
      if(theL1CentralJetCollection.size()>0) {
	hL1CentralJet1PtAfterL1[i]->Fill(theL1CentralJetCollection[0].pt());
	hL1CentralJet1EtaAfterL1[i]->Fill(theL1CentralJetCollection[0].eta());
	hL1CentralJet1PhiAfterL1[i]->Fill(theL1CentralJetCollection[0].phi());
      }
      if(theL1CentralJetCollection.size()>1) {
	hL1CentralJet2PtAfterL1[i]->Fill(theL1CentralJetCollection[1].pt());
	hL1CentralJet2EtaAfterL1[i]->Fill(theL1CentralJetCollection[1].eta());
	hL1CentralJet2PhiAfterL1[i]->Fill(theL1CentralJetCollection[1].phi());
      }
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1CentralJetMultAfterHLT[i]->Fill(theL1CentralJetCollection.size());
      if(theL1CentralJetCollection.size()>0) {
	hL1CentralJet1PtAfterHLT[i]->Fill(theL1CentralJetCollection[0].pt());
	hL1CentralJet1EtaAfterHLT[i]->Fill(theL1CentralJetCollection[0].eta());
	hL1CentralJet1PhiAfterHLT[i]->Fill(theL1CentralJetCollection[0].phi());
      }
      if(theL1CentralJetCollection.size()>1) {
	hL1CentralJet2PtAfterHLT[i]->Fill(theL1CentralJetCollection[1].pt());
	hL1CentralJet2EtaAfterHLT[i]->Fill(theL1CentralJetCollection[1].eta());
	hL1CentralJet2PhiAfterHLT[i]->Fill(theL1CentralJetCollection[1].phi());
      }
    }
  }





  hL1ForwardJetMult->Fill(theL1ForwardJetCollection.size());
  if(theL1ForwardJetCollection.size()>0) {
    hL1ForwardJet1Pt->Fill(theL1ForwardJetCollection[0].pt());
    hL1ForwardJet1Eta->Fill(theL1ForwardJetCollection[0].eta());
    hL1ForwardJet1Phi->Fill(theL1ForwardJetCollection[0].phi());
  }
  if(theL1ForwardJetCollection.size()>1) {
    hL1ForwardJet2Pt->Fill(theL1ForwardJetCollection[1].pt());
    hL1ForwardJet2Eta->Fill(theL1ForwardJetCollection[1].eta());
    hL1ForwardJet2Phi->Fill(theL1ForwardJetCollection[1].phi());
  }
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1ForwardJetMultAfterL1[i]->Fill(theL1ForwardJetCollection.size());
      if(theL1ForwardJetCollection.size()>0) {
	hL1ForwardJet1PtAfterL1[i]->Fill(theL1ForwardJetCollection[0].pt());
	hL1ForwardJet1EtaAfterL1[i]->Fill(theL1ForwardJetCollection[0].eta());
	hL1ForwardJet1PhiAfterL1[i]->Fill(theL1ForwardJetCollection[0].phi());
      }
      if(theL1ForwardJetCollection.size()>1) {
	hL1ForwardJet2PtAfterL1[i]->Fill(theL1ForwardJetCollection[1].pt());
	hL1ForwardJet2EtaAfterL1[i]->Fill(theL1ForwardJetCollection[1].eta());
	hL1ForwardJet2PhiAfterL1[i]->Fill(theL1ForwardJetCollection[1].phi());
      }
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1ForwardJetMultAfterHLT[i]->Fill(theL1ForwardJetCollection.size());
      if(theL1ForwardJetCollection.size()>0) {
	hL1ForwardJet1PtAfterHLT[i]->Fill(theL1ForwardJetCollection[0].pt());
	hL1ForwardJet1EtaAfterHLT[i]->Fill(theL1ForwardJetCollection[0].eta());
	hL1ForwardJet1PhiAfterHLT[i]->Fill(theL1ForwardJetCollection[0].phi());
      }
      if(theL1ForwardJetCollection.size()>1) {
	hL1ForwardJet2PtAfterHLT[i]->Fill(theL1ForwardJetCollection[1].pt());
	hL1ForwardJet2EtaAfterHLT[i]->Fill(theL1ForwardJetCollection[1].eta());
	hL1ForwardJet2PhiAfterHLT[i]->Fill(theL1ForwardJetCollection[1].phi());
      }
    }
  }





  hL1TauJetMult->Fill(theL1TauJetCollection.size());
  if(theL1TauJetCollection.size()>0) {
    hL1TauJet1Pt->Fill(theL1TauJetCollection[0].pt());
    hL1TauJet1Eta->Fill(theL1TauJetCollection[0].eta());
    hL1TauJet1Phi->Fill(theL1TauJetCollection[0].phi());
  }
  if(theL1TauJetCollection.size()>1) {
    hL1TauJet2Pt->Fill(theL1TauJetCollection[1].pt());
    hL1TauJet2Eta->Fill(theL1TauJetCollection[1].eta());
    hL1TauJet2Phi->Fill(theL1TauJetCollection[1].phi());
  }

  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1TauJetMultAfterL1[i]->Fill(theL1TauJetCollection.size());
      if(theL1TauJetCollection.size()>0) {
	hL1TauJet1PtAfterL1[i]->Fill(theL1TauJetCollection[0].pt());
	hL1TauJet1EtaAfterL1[i]->Fill(theL1TauJetCollection[0].eta());
	hL1TauJet1PhiAfterL1[i]->Fill(theL1TauJetCollection[0].phi());
      }
      if(theL1TauJetCollection.size()>1) {
	hL1TauJet2PtAfterL1[i]->Fill(theL1TauJetCollection[1].pt());
	hL1TauJet2EtaAfterL1[i]->Fill(theL1TauJetCollection[1].eta());
	hL1TauJet2PhiAfterL1[i]->Fill(theL1TauJetCollection[1].phi());
      }
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1TauJetMultAfterHLT[i]->Fill(theL1TauJetCollection.size());
      if(theL1TauJetCollection.size()>0) {
	hL1TauJet1PtAfterHLT[i]->Fill(theL1TauJetCollection[0].pt());
	hL1TauJet1EtaAfterHLT[i]->Fill(theL1TauJetCollection[0].eta());
	hL1TauJet1PhiAfterHLT[i]->Fill(theL1TauJetCollection[0].phi());
      }
      if(theL1TauJetCollection.size()>1) {
	hL1TauJet2PtAfterHLT[i]->Fill(theL1TauJetCollection[1].pt());
	hL1TauJet2EtaAfterHLT[i]->Fill(theL1TauJetCollection[1].eta());
	hL1TauJet2PhiAfterHLT[i]->Fill(theL1TauJetCollection[1].phi());
      }
    }
  }




  hL1EmIsoMult->Fill(theL1EmIsoCollection.size());
  if(theL1EmIsoCollection.size()>0) {
    hL1EmIso1Pt->Fill(theL1EmIsoCollection[0].pt());
    hL1EmIso1Eta->Fill(theL1EmIsoCollection[0].eta());
    hL1EmIso1Phi->Fill(theL1EmIsoCollection[0].phi());
  }
  if(theL1EmIsoCollection.size()>1) {
    hL1EmIso2Pt->Fill(theL1EmIsoCollection[1].pt());
    hL1EmIso2Eta->Fill(theL1EmIsoCollection[1].eta());
    hL1EmIso2Phi->Fill(theL1EmIsoCollection[1].phi());
  }
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1EmIsoMultAfterL1[i]->Fill(theL1EmIsoCollection.size());
      if(theL1EmIsoCollection.size()>0) {
	hL1EmIso1PtAfterL1[i]->Fill(theL1EmIsoCollection[0].pt());
	hL1EmIso1EtaAfterL1[i]->Fill(theL1EmIsoCollection[0].eta());
	hL1EmIso1PhiAfterL1[i]->Fill(theL1EmIsoCollection[0].phi());
      }
      if(theL1EmIsoCollection.size()>1) {
	hL1EmIso2PtAfterL1[i]->Fill(theL1EmIsoCollection[1].pt());
	hL1EmIso2EtaAfterL1[i]->Fill(theL1EmIsoCollection[1].eta());
	hL1EmIso2PhiAfterL1[i]->Fill(theL1EmIsoCollection[1].phi());
      }
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1EmIsoMultAfterHLT[i]->Fill(theL1EmIsoCollection.size());
      if(theL1EmIsoCollection.size()>0) {
	hL1EmIso1PtAfterHLT[i]->Fill(theL1EmIsoCollection[0].pt());
	hL1EmIso1EtaAfterHLT[i]->Fill(theL1EmIsoCollection[0].eta());
	hL1EmIso1PhiAfterHLT[i]->Fill(theL1EmIsoCollection[0].phi());
      }
      if(theL1EmIsoCollection.size()>1) {
	hL1EmIso2PtAfterHLT[i]->Fill(theL1EmIsoCollection[1].pt());
	hL1EmIso2EtaAfterHLT[i]->Fill(theL1EmIsoCollection[1].eta());
	hL1EmIso2PhiAfterHLT[i]->Fill(theL1EmIsoCollection[1].phi());
      }
    }
  }



  hL1EmNotIsoMult->Fill(theL1EmNotIsoCollection.size());
  if(theL1EmNotIsoCollection.size()>0) {
    hL1EmNotIso1Pt->Fill(theL1EmNotIsoCollection[0].pt());
    hL1EmNotIso1Eta->Fill(theL1EmNotIsoCollection[0].eta());
    hL1EmNotIso1Phi->Fill(theL1EmNotIsoCollection[0].phi());
  }
  if(theL1EmNotIsoCollection.size()>1) {
    hL1EmNotIso2Pt->Fill(theL1EmNotIsoCollection[1].pt());
    hL1EmNotIso2Eta->Fill(theL1EmNotIsoCollection[1].eta());
    hL1EmNotIso2Phi->Fill(theL1EmNotIsoCollection[1].phi());
  }
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1EmNotIsoMultAfterL1[i]->Fill(theL1EmNotIsoCollection.size());
      if(theL1EmNotIsoCollection.size()>0) {
	hL1EmNotIso1PtAfterL1[i]->Fill(theL1EmNotIsoCollection[0].pt());
	hL1EmNotIso1EtaAfterL1[i]->Fill(theL1EmNotIsoCollection[0].eta());
	hL1EmNotIso1PhiAfterL1[i]->Fill(theL1EmNotIsoCollection[0].phi());
      }
      if(theL1EmNotIsoCollection.size()>1) {
	hL1EmNotIso2PtAfterL1[i]->Fill(theL1EmNotIsoCollection[1].pt());
	hL1EmNotIso2EtaAfterL1[i]->Fill(theL1EmNotIsoCollection[1].eta());
	hL1EmNotIso2PhiAfterL1[i]->Fill(theL1EmNotIsoCollection[1].phi());
      }
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1EmNotIsoMultAfterHLT[i]->Fill(theL1EmNotIsoCollection.size());
      if(theL1EmNotIsoCollection.size()>0) {
	hL1EmNotIso1PtAfterHLT[i]->Fill(theL1EmNotIsoCollection[0].pt());
	hL1EmNotIso1EtaAfterHLT[i]->Fill(theL1EmNotIsoCollection[0].eta());
	hL1EmNotIso1PhiAfterHLT[i]->Fill(theL1EmNotIsoCollection[0].phi());
      }
      if(theL1EmNotIsoCollection.size()>1) {
	hL1EmNotIso2PtAfterHLT[i]->Fill(theL1EmNotIsoCollection[1].pt());
	hL1EmNotIso2EtaAfterHLT[i]->Fill(theL1EmNotIsoCollection[1].eta());
	hL1EmNotIso2PhiAfterHLT[i]->Fill(theL1EmNotIsoCollection[1].phi());
      }
    }
  }




  //**********************
  // Fill the Muon Histos
  //**********************
  
  hL1MuonMult->Fill(theL1MuonCollection.size());
  if(theL1MuonCollection.size()>0) {
    hL1Muon1Pt->Fill(theL1MuonCollection[0].pt());
    hL1Muon1Eta->Fill(theL1MuonCollection[0].eta());
    hL1Muon1Phi->Fill(theL1MuonCollection[0].phi());
  }
  if(theL1MuonCollection.size()>1) {
    hL1Muon2Pt->Fill(theL1MuonCollection[1].pt());
    hL1Muon2Eta->Fill(theL1MuonCollection[1].eta());
    hL1Muon2Phi->Fill(theL1MuonCollection[1].phi());
  }
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1MuonMultAfterL1[i]->Fill(theL1MuonCollection.size());
      if(theL1MuonCollection.size()>0) {
	hL1Muon1PtAfterL1[i]->Fill(theL1MuonCollection[0].pt());
	hL1Muon1EtaAfterL1[i]->Fill(theL1MuonCollection[0].eta());
	hL1Muon1PhiAfterL1[i]->Fill(theL1MuonCollection[0].phi());
      }
      if(theL1MuonCollection.size()>1) {
	hL1Muon2PtAfterL1[i]->Fill(theL1MuonCollection[1].pt());
	hL1Muon2EtaAfterL1[i]->Fill(theL1MuonCollection[1].eta());
	hL1Muon2PhiAfterL1[i]->Fill(theL1MuonCollection[1].phi());
      }
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1MuonMultAfterHLT[i]->Fill(theL1MuonCollection.size());
      if(theL1MuonCollection.size()>0) {
	hL1Muon1PtAfterHLT[i]->Fill(theL1MuonCollection[0].pt());
	hL1Muon1EtaAfterHLT[i]->Fill(theL1MuonCollection[0].eta());
	hL1Muon1PhiAfterHLT[i]->Fill(theL1MuonCollection[0].phi());
      }
      if(theL1MuonCollection.size()>1) {
	hL1Muon2PtAfterHLT[i]->Fill(theL1MuonCollection[1].pt());
	hL1Muon2EtaAfterHLT[i]->Fill(theL1MuonCollection[1].eta());
	hL1Muon2PhiAfterHLT[i]->Fill(theL1MuonCollection[1].phi());
      }
    }
  }

  //**********************
  // Fill the MET Histos
  //**********************
  
  hL1MET->Fill(theL1METCollection[0].etMiss());
  hL1METx->Fill(theL1METCollection[0].px());
  hL1METy->Fill(theL1METCollection[0].py());
  hL1METphi->Fill(theL1METCollection[0].phi());
  hL1SumEt->Fill(theL1METCollection[0].etTotal());
  double L1MetSig = theL1METCollection[0].etMiss() / sqrt(theL1METCollection[0].etTotal());
  hL1METSignificance->Fill(L1MetSig);
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1METAfterL1[i]->Fill(theL1METCollection[0].etMiss());
      hL1METxAfterL1[i]->Fill(theL1METCollection[0].px());
      hL1METyAfterL1[i]->Fill(theL1METCollection[0].py());
      hL1METphiAfterL1[i]->Fill(theL1METCollection[0].phi());
      hL1SumEtAfterL1[i]->Fill(theL1METCollection[0].etTotal());
      hL1METSignificanceAfterL1[i]->Fill(L1MetSig);
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1METAfterHLT[i]->Fill(theL1METCollection[0].etMiss());
      hL1METxAfterHLT[i]->Fill(theL1METCollection[0].px());
      hL1METyAfterHLT[i]->Fill(theL1METCollection[0].py());
      hL1METphiAfterHLT[i]->Fill(theL1METCollection[0].phi());
      hL1SumEtAfterHLT[i]->Fill(theL1METCollection[0].etTotal());
      hL1METSignificanceAfterHLT[i]->Fill(L1MetSig);
    }
  }

  //**********************
  // Fill the Reco Object Histos
  //**********************
  //**********************
  // Fill the Jet Histos
  //**********************
  
  int nJets = 0;
  std::vector<double> diJetInvMass;
  for(unsigned int i=0; i<theCaloJetCollection.size(); i++) {
    if(theCaloJetCollection[i].pt() > def_jetPtMin ) {
      nJets++;
      for(unsigned int j=i+1; j<theCaloJetCollection.size(); j++) {
	if(theCaloJetCollection[j].pt() > def_jetPtMin ) {
	  diJetInvMass.push_back(invariantMass(&theCaloJetCollection[i],&theCaloJetCollection[j]));
	}
      }
    }
  } 
  
  hJetMult->Fill(nJets);
  for(unsigned int j=0; j<diJetInvMass.size(); j++) {hDiJetInvMass->Fill(diJetInvMass[j]);}
  if(theCaloJetCollection.size()>0) {
    hJet1Pt->Fill(theCaloJetCollection[0].pt());
    hJet1Eta->Fill(theCaloJetCollection[0].eta());
    hJet1Phi->Fill(theCaloJetCollection[0].phi());
  }
  if(theCaloJetCollection.size()>1) {
    hJet2Pt->Fill(theCaloJetCollection[1].pt());
    hJet2Eta->Fill(theCaloJetCollection[1].eta());
    hJet2Phi->Fill(theCaloJetCollection[1].phi());
  }
  //  for(int i=0; i<theCaloJetCollection.size(); i++) cout << "theCaloJetCollection)[0].pt = " << theCaloJetCollection[i].pt() << endl;
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hJetMultAfterL1[i]->Fill(nJets);
      for(unsigned int j=0; j<diJetInvMass.size(); j++) {hDiJetInvMassAfterL1[i]->Fill(diJetInvMass[j]);}
      if(theCaloJetCollection.size()>0) {
	hJet1PtAfterL1[i]->Fill(theCaloJetCollection[0].pt());
	hJet1EtaAfterL1[i]->Fill(theCaloJetCollection[0].eta());
	hJet1PhiAfterL1[i]->Fill(theCaloJetCollection[0].phi());
      }
      if(theCaloJetCollection.size()>1) {
	hJet2PtAfterL1[i]->Fill(theCaloJetCollection[1].pt());
	hJet2EtaAfterL1[i]->Fill(theCaloJetCollection[1].eta());
	hJet2PhiAfterL1[i]->Fill(theCaloJetCollection[1].phi());
      }
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hJetMultAfterHLT[i]->Fill(nJets);
      for(unsigned int j=0; j<diJetInvMass.size(); j++) {hDiJetInvMassAfterHLT[i]->Fill(diJetInvMass[j]);}
      if(theCaloJetCollection.size()>0) {
	hJet1PtAfterHLT[i]->Fill(theCaloJetCollection[0].pt());
	hJet1EtaAfterHLT[i]->Fill(theCaloJetCollection[0].eta());
	hJet1PhiAfterHLT[i]->Fill(theCaloJetCollection[0].phi());
      }
      if(theCaloJetCollection.size()>1) {
	hJet2PtAfterHLT[i]->Fill(theCaloJetCollection[1].pt());
	hJet2EtaAfterHLT[i]->Fill(theCaloJetCollection[1].eta());
	hJet2PhiAfterHLT[i]->Fill(theCaloJetCollection[1].phi());
      }
    }
  }


  //**********************
  // Fill the Electron Histos
  //**********************
  
  int nElectrons = 0;
  std::vector<double> diElecInvMass;
  for(unsigned int i=0; i<theElectronCollection.size(); i++) {
    if(theElectronCollection[i].pt() > def_electronPtMin ) {
      nElectrons++;
      for(unsigned int j=i+1; j<theElectronCollection.size(); j++) {
	if(theElectronCollection[j].pt() > def_electronPtMin ) {
	  if(theElectronCollection[i].charge()*theElectronCollection[j].charge() < 0)
	    diElecInvMass.push_back(invariantMass(&theElectronCollection[i],&theElectronCollection[j]));
	}
      }
    }
  }

  hElecMult->Fill(nElectrons);
  for(unsigned int j=0; j<diElecInvMass.size(); j++) {hDiElecInvMass->Fill(diElecInvMass[j]);}
  if(theElectronCollection.size()>0) {
    hElec1Pt->Fill(theElectronCollection[0].pt());
    hElec1Eta->Fill(theElectronCollection[0].eta());
    hElec1Phi->Fill(theElectronCollection[0].phi());
  }
  if(theElectronCollection.size()>1) {
    hElec2Pt->Fill(theElectronCollection[1].pt());
    hElec2Eta->Fill(theElectronCollection[1].eta());
    hElec2Phi->Fill(theElectronCollection[1].phi());
  }
  //  for(int i=0; i<theElectronCollection.size(); i++) cout << "(*)theElectronCollection[0].pt = " << theElectronCollection[i].pt() << endl;
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hElecMultAfterL1[i]->Fill(nElectrons);
      for(unsigned int j=0; j<diElecInvMass.size(); j++) {hDiElecInvMassAfterL1[i]->Fill(diElecInvMass[j]);}
      if(theElectronCollection.size()>0) {
	hElec1PtAfterL1[i]->Fill(theElectronCollection[0].pt());
	hElec1EtaAfterL1[i]->Fill(theElectronCollection[0].eta());
	hElec1PhiAfterL1[i]->Fill(theElectronCollection[0].phi());
      }
      if(theElectronCollection.size()>1) {
	hElec2PtAfterL1[i]->Fill(theElectronCollection[1].pt());
	hElec2EtaAfterL1[i]->Fill(theElectronCollection[1].eta());
	hElec2PhiAfterL1[i]->Fill(theElectronCollection[1].phi());
      }
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hElecMultAfterHLT[i]->Fill(nElectrons);
      for(unsigned int j=0; j<diElecInvMass.size(); j++) {hDiElecInvMassAfterHLT[i]->Fill(diElecInvMass[j]);}
      if(theElectronCollection.size()>0) {
	hElec1PtAfterHLT[i]->Fill(theElectronCollection[0].pt());
	hElec1EtaAfterHLT[i]->Fill(theElectronCollection[0].eta());
	hElec1PhiAfterHLT[i]->Fill(theElectronCollection[0].phi());
      }
      if(theElectronCollection.size()>1) {
	hElec2PtAfterHLT[i]->Fill(theElectronCollection[1].pt());
	hElec2EtaAfterHLT[i]->Fill(theElectronCollection[1].eta());
	hElec2PhiAfterHLT[i]->Fill(theElectronCollection[1].phi());
      }
    }
  }


  //**********************
  // Fill the Muon Histos
  //**********************
  
  int nMuons = 0;
  std::vector<double> diMuonInvMass;
  for(unsigned int i=0; i<theMuonCollection.size(); i++) {
    if(theMuonCollection[i].pt() > def_muonPtMin ) {
      nMuons++;
      for(unsigned int j=i+1; j<theMuonCollection.size(); j++) {
	if(theMuonCollection[j].pt() > def_muonPtMin ) {
	  if(theMuonCollection[i].charge()*theMuonCollection[j].charge() < 0)
	    diMuonInvMass.push_back(invariantMass(&theMuonCollection[i],&theMuonCollection[j]));
	}
      }
    }
  }


  hMuonMult->Fill(nMuons);
  for(unsigned int j=0; j<diMuonInvMass.size(); j++) {hDiMuonInvMass->Fill(diMuonInvMass[j]);}
  if(theMuonCollection.size()>0) {
    hMuon1Pt->Fill(theMuonCollection[0].pt());
    hMuon1Eta->Fill(theMuonCollection[0].eta());
    hMuon1Phi->Fill(theMuonCollection[0].phi());
  }
  if(theMuonCollection.size()>1) {
    hMuon2Pt->Fill(theMuonCollection[1].pt());
    hMuon2Eta->Fill(theMuonCollection[1].eta());
    hMuon2Phi->Fill(theMuonCollection[1].phi());
  }
  //  for(int i=0; i<theMuonCollection.size(); i++) cout << "theMuonCollection)[0].pt = " << theMuonCollection[i].pt() << endl;
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hMuonMultAfterL1[i]->Fill(nMuons);
      for(unsigned int j=0; j<diMuonInvMass.size(); j++) {hDiMuonInvMassAfterL1[i]->Fill(diMuonInvMass[j]);}
      if(theMuonCollection.size()>0) {
	hMuon1PtAfterL1[i]->Fill(theMuonCollection[0].pt());
	hMuon1EtaAfterL1[i]->Fill(theMuonCollection[0].eta());
	hMuon1PhiAfterL1[i]->Fill(theMuonCollection[0].phi());
      }
      if(theMuonCollection.size()>1) {
	hMuon2PtAfterL1[i]->Fill(theMuonCollection[1].pt());
	hMuon2EtaAfterL1[i]->Fill(theMuonCollection[1].eta());
	hMuon2PhiAfterL1[i]->Fill(theMuonCollection[1].phi());
      }
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hMuonMultAfterHLT[i]->Fill(nMuons);
      for(unsigned int j=0; j<diMuonInvMass.size(); j++) {hDiMuonInvMassAfterHLT[i]->Fill(diMuonInvMass[j]);}
      if(theMuonCollection.size()>0) {
	hMuon1PtAfterHLT[i]->Fill(theMuonCollection[0].pt());
	hMuon1EtaAfterHLT[i]->Fill(theMuonCollection[0].eta());
 	hMuon1PhiAfterHLT[i]->Fill(theMuonCollection[0].phi());
      }
      if(theMuonCollection.size()>1) {
	hMuon2PtAfterHLT[i]->Fill(theMuonCollection[1].pt());
	hMuon2EtaAfterHLT[i]->Fill(theMuonCollection[1].eta());
	hMuon2PhiAfterHLT[i]->Fill(theMuonCollection[1].phi());
      }
    }
  }

  //**********************
  // Fill the Photon Histos
  //**********************
  
  int nPhotons = 0;
  std::vector<double> diPhotonInvMass;
  for(unsigned int i=0; i<thePhotonCollection.size(); i++) {
    if(thePhotonCollection[i].pt() > def_photonPtMin ) {
      nPhotons++;
      for(unsigned int j=i+1; j<thePhotonCollection.size(); j++) {
	if(thePhotonCollection[j].pt() > def_photonPtMin ) {
	  diPhotonInvMass.push_back(invariantMass(&thePhotonCollection[i],&thePhotonCollection[j]));
	}
      }
    }
  }

  hPhotonMult->Fill(nPhotons);
  for(unsigned int j=0; j<diPhotonInvMass.size(); j++) {hDiPhotonInvMass->Fill(diPhotonInvMass[j]);}
  if(thePhotonCollection.size()>0) {
    hPhoton1Pt->Fill(thePhotonCollection[0].et());
    hPhoton1Eta->Fill(thePhotonCollection[0].eta());
    hPhoton1Phi->Fill(thePhotonCollection[0].phi());
  }
  if(thePhotonCollection.size()>1) {
    hPhoton2Pt->Fill(thePhotonCollection[1].et());
    hPhoton2Eta->Fill(thePhotonCollection[1].eta());
    hPhoton2Phi->Fill(thePhotonCollection[1].phi());
  }
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hPhotonMultAfterL1[i]->Fill(nPhotons);
      for(unsigned int j=0; j<diPhotonInvMass.size(); j++) {hDiPhotonInvMassAfterL1[i]->Fill(diPhotonInvMass[j]);}
      if(thePhotonCollection.size()>0) {
	hPhoton1PtAfterL1[i]->Fill(thePhotonCollection[0].et());
	hPhoton1EtaAfterL1[i]->Fill(thePhotonCollection[0].eta());
	hPhoton1PhiAfterL1[i]->Fill(thePhotonCollection[0].phi());
      }
      if(thePhotonCollection.size()>1) {
	hPhoton2PtAfterL1[i]->Fill(thePhotonCollection[1].et());
	hPhoton2EtaAfterL1[i]->Fill(thePhotonCollection[1].eta());
	hPhoton2PhiAfterL1[i]->Fill(thePhotonCollection[1].phi());
      }
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hPhotonMultAfterHLT[i]->Fill(nPhotons);
      for(unsigned int j=0; j<diPhotonInvMass.size(); j++) {hDiPhotonInvMassAfterHLT[i]->Fill(diPhotonInvMass[j]);}
      if(thePhotonCollection.size()>0) {
	hPhoton1PtAfterHLT[i]->Fill(thePhotonCollection[0].et());
	hPhoton1EtaAfterHLT[i]->Fill(thePhotonCollection[0].eta());
	hPhoton1PhiAfterHLT[i]->Fill(thePhotonCollection[0].phi());
      }
      if(thePhotonCollection.size()>1) {
	hPhoton2PtAfterHLT[i]->Fill(thePhotonCollection[1].et());
	hPhoton2EtaAfterHLT[i]->Fill(thePhotonCollection[1].eta());
 	hPhoton2PhiAfterHLT[i]->Fill(thePhotonCollection[1].phi());
     }
    }
  }


  //**********************
  // Fill the MET Histos
  //**********************
  
  hMET->Fill((theCaloMETCollection.front()).pt());
  hMETx->Fill((theCaloMETCollection.front()).px());
  hMETy->Fill((theCaloMETCollection.front()).py());
  hMETphi->Fill((theCaloMETCollection.front()).phi());
  hSumEt->Fill((theCaloMETCollection.front()).sumEt());
  double RecoMetSig = (theCaloMETCollection.front()).pt() / sqrt( (theCaloMETCollection.front()).sumEt() );
  hMETSignificance->Fill(RecoMetSig);
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hMETAfterL1[i]->Fill((theCaloMETCollection.front()).pt());
      hMETxAfterL1[i]->Fill((theCaloMETCollection.front()).px());
      hMETyAfterL1[i]->Fill((theCaloMETCollection.front()).py());
      hMETphiAfterL1[i]->Fill((theCaloMETCollection.front()).phi());
      hSumEtAfterL1[i]->Fill((theCaloMETCollection.front()).sumEt());
      hMETSignificanceAfterL1[i]->Fill(RecoMetSig);
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hMETAfterHLT[i]->Fill((theCaloMETCollection.front()).pt());
      hMETxAfterHLT[i]->Fill((theCaloMETCollection.front()).px());
      hMETyAfterHLT[i]->Fill((theCaloMETCollection.front()).py());
      hMETphiAfterHLT[i]->Fill((theCaloMETCollection.front()).phi());
      hSumEtAfterHLT[i]->Fill((theCaloMETCollection.front()).sumEt());
      hMETSignificanceAfterHLT[i]->Fill(RecoMetSig);
    }
  }


}




void PlotMaker::bookHistos(std::vector<int>* l1bits, std::vector<int>* hltbits, 
			   std::vector<std::string>* l1Names_, std::vector<std::string>* hlNames_)
{

  this->setBits(l1bits, hltbits);

  //******************
  //Book histos for L1 Objects
  //******************
  
  
  //******************
  //Book Jets
  //******************

  gDirectory->cd("/L1Jets/Central/General");
  hL1CentralJetMult = new TH1D("JetMult", "Jet Multiplicity", 10, 0, 10);
  hL1CentralJet1Pt  = new TH1D("Jet1Pt",  "Jet 1 Pt ",        100, 0, 1000);
  hL1CentralJet2Pt  = new TH1D("Jet2Pt",  "Jet 2 Pt ",        100, 0, 1000);
  hL1CentralJet1Eta  = new TH1D("Jet1Eta",  "Jet 1 Eta ",     100, -3, 3);
  hL1CentralJet2Eta  = new TH1D("Jet2Eta",  "Jet 2 Eta ",     100, -3, 3);
  hL1CentralJet1Phi  = new TH1D("Jet1Phi",  "Jet 1 Phi ",     100, -3.2, 3.2);
  hL1CentralJet2Phi  = new TH1D("Jet2Phi",  "Jet 2 Phi ",     100, -3.2, 3.2);

  gDirectory->cd("/L1Jets/Central/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "JetMult_" + (*l1Names_)[i];
    myHistoTitle = "Jet Multiplicity for L1 path " + (*l1Names_)[i];
    hL1CentralJetMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Pt for L1 path " + (*l1Names_)[i];
    hL1CentralJet1PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Pt for L1 path " + (*l1Names_)[i];
    hL1CentralJet2PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Eta for L1 path " + (*l1Names_)[i];
    hL1CentralJet1EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Eta for L1 path " + (*l1Names_)[i];
    hL1CentralJet2EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Phi for L1 path " + (*l1Names_)[i];
   hL1CentralJet1PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Phi for L1 path " + (*l1Names_)[i];
    hL1CentralJet2PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  gDirectory->cd("/L1Jets/Central/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "JetMult_" + (*hlNames_)[i];
    myHistoTitle = "Jet Multiplicity for HLT path " + (*hlNames_)[i];
    hL1CentralJetMultAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Pt for HLT path " + (*hlNames_)[i];
    hL1CentralJet1PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Pt for HLT path " + (*hlNames_)[i];
    hL1CentralJet2PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Eta for HLT path " + (*hlNames_)[i];
    hL1CentralJet1EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Eta for HLT path " + (*hlNames_)[i];
    hL1CentralJet2EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Phi for HLT path " + (*hlNames_)[i];
    hL1CentralJet1PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Phi for HLT path " + (*hlNames_)[i];
    hL1CentralJet2PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  gDirectory->cd("/L1Jets/Forward/General");
  hL1ForwardJetMult = new TH1D("JetMult", "Jet Multiplicity", 10, 0, 10);
  hL1ForwardJet1Pt  = new TH1D("Jet1Pt",  "Jet 1 Pt ",        100, 0, 1000);
  hL1ForwardJet2Pt  = new TH1D("Jet2Pt",  "Jet 2 Pt ",        100, 0, 1000);
  hL1ForwardJet1Eta  = new TH1D("Jet1Eta",  "Jet 1 Eta ",        100, -3, 3);
  hL1ForwardJet2Eta  = new TH1D("Jet2Eta",  "Jet 2 Eta ",        100, -3, 3);
  hL1ForwardJet1Phi  = new TH1D("Jet1Phi",  "Jet 1 Phi ",        100, -3.2, 3.2);
  hL1ForwardJet2Phi  = new TH1D("Jet2Phi",  "Jet 2 Phi ",        100, -3.2, 3.2);

  gDirectory->cd("/L1Jets/Forward/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "JetMult_" + (*l1Names_)[i];
    myHistoTitle = "Jet Multiplicity for L1 path " + (*l1Names_)[i];
    hL1ForwardJetMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Pt for L1 path " + (*l1Names_)[i];
    hL1ForwardJet1PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Pt for L1 path " + (*l1Names_)[i];
    hL1ForwardJet2PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Eta for L1 path " + (*l1Names_)[i];
    hL1ForwardJet1EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Eta for L1 path " + (*l1Names_)[i];
    hL1ForwardJet2EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Phi for L1 path " + (*l1Names_)[i];
    hL1ForwardJet1PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Phi for L1 path " + (*l1Names_)[i];
    hL1ForwardJet2PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  gDirectory->cd("/L1Jets/Forward/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "JetMult_" + (*hlNames_)[i];
    myHistoTitle = "Jet Multiplicity for HLT path " + (*hlNames_)[i];
    hL1ForwardJetMultAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Pt for HLT path " + (*hlNames_)[i];
    hL1ForwardJet1PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Pt for HLT path " + (*hlNames_)[i];
    hL1ForwardJet2PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Eta for HLT path " + (*hlNames_)[i];
    hL1ForwardJet1EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Eta for HLT path " + (*hlNames_)[i];
    hL1ForwardJet2EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Phi for HLT path " + (*hlNames_)[i];
    hL1ForwardJet1PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Phi for HLT path " + (*hlNames_)[i];
    hL1ForwardJet2PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  gDirectory->cd("/L1Jets/Tau/General");
  hL1TauJetMult = new TH1D("JetMult", "Jet Multiplicity", 10, 0, 10);
  hL1TauJet1Pt  = new TH1D("Jet1Pt",  "Jet 1 Pt ",        100, 0, 1000);
  hL1TauJet2Pt  = new TH1D("Jet2Pt",  "Jet 2 Pt ",        100, 0, 1000);
  hL1TauJet1Eta  = new TH1D("Jet1Eta",  "Jet 1 Eta ",        100, -3, -3);
  hL1TauJet2Eta  = new TH1D("Jet2Eta",  "Jet 2 Eta ",        100, -3, -3);
  hL1TauJet1Phi  = new TH1D("Jet1Phi",  "Jet 1 Phi ",        100, -3.2, -3.2);
  hL1TauJet2Phi  = new TH1D("Jet2Phi",  "Jet 2 Phi ",        100, -3.2, -3.2);

  gDirectory->cd("/L1Jets/Tau/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "JetMult_" + (*l1Names_)[i];
    myHistoTitle = "Jet Multiplicity for L1 path " + (*l1Names_)[i];
    hL1TauJetMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Pt for L1 path " + (*l1Names_)[i];
    hL1TauJet1PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Pt for L1 path " + (*l1Names_)[i];
    hL1TauJet2PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Eta for L1 path " + (*l1Names_)[i];
    hL1TauJet1EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Eta for L1 path " + (*l1Names_)[i];
    hL1TauJet2EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Phi for L1 path " + (*l1Names_)[i];
    hL1TauJet1PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Phi for L1 path " + (*l1Names_)[i];
    hL1TauJet2PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  gDirectory->cd("/L1Jets/Tau/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "JetMult_" + (*hlNames_)[i];
    myHistoTitle = "Jet Multiplicity for HLT path " + (*l1Names_)[i];
    hL1TauJetMultAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Pt for HLT path " + (*l1Names_)[i];
    hL1TauJet1PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Pt for HLT path " + (*l1Names_)[i];
    hL1TauJet2PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Eta for HLT path " + (*l1Names_)[i];
    hL1TauJet1EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Eta for HLT path " + (*l1Names_)[i];
    hL1TauJet2EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Phi for HLT path " + (*l1Names_)[i];
    hL1TauJet1PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Phi for HLT path " + (*l1Names_)[i];
    hL1TauJet2PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }



  gDirectory->cd("/L1Em/Isolated/General");
  hL1EmIsoMult = new TH1D("ElecMult", "Elec Multiplicity", 10, 0, 10);
  hL1EmIso1Pt  = new TH1D("Elec1Pt",  "Elec 1 Pt ",        100, 0, 100);
  hL1EmIso2Pt  = new TH1D("Elec2Pt",  "Elec 2 Pt ",        100, 0, 100);
  hL1EmIso1Eta  = new TH1D("Elec1Eta",  "Elec 1 Eta ",        100, -3, 3);
  hL1EmIso2Eta  = new TH1D("Elec2Eta",  "Elec 2 Eta ",        100, -3, 3);
  hL1EmIso1Phi  = new TH1D("Elec1Phi",  "Elec 1 Phi ",        100, -3.2, 3.2);
  hL1EmIso2Phi  = new TH1D("Elec2Phi",  "Elec 2 Phi ",        100, -3.2, 3.2);
  
  gDirectory->cd("/L1Em/Isolated/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "ElecMult_" + (*l1Names_)[i];
    myHistoTitle = "Elec Multiplicity for L1 path " + (*l1Names_)[i];
    hL1EmIsoMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Elec1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Pt for L1 path " + (*l1Names_)[i];
    hL1EmIso1PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Pt for L1 path " + (*l1Names_)[i];
    hL1EmIso2PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Eta for L1 path " + (*l1Names_)[i];
    hL1EmIso1EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Eta for L1 path " + (*l1Names_)[i];
    hL1EmIso2EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Phi for L1 path " + (*l1Names_)[i];
    hL1EmIso1PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Elec2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Phi for L1 path " + (*l1Names_)[i];
    hL1EmIso2PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  gDirectory->cd("/L1Em/Isolated/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "ElecMult_" + (*hlNames_)[i];
    myHistoTitle = "Elec Multiplicity for HLT path " + (*hlNames_)[i];    
    hL1EmIsoMultAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Elec1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Pt for HLT path " + (*hlNames_)[i];
    hL1EmIso1PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Pt for HLT path " + (*hlNames_)[i];
    hL1EmIso2PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Eta for HLT path " + (*hlNames_)[i];
    hL1EmIso1EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Eta for HLT path " + (*hlNames_)[i];
    hL1EmIso2EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Phi for HLT path " + (*hlNames_)[i];
    hL1EmIso1PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Elec2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Phi for HLT path " + (*hlNames_)[i];
    hL1EmIso2PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }
  gDirectory->cd();



  gDirectory->cd("/L1Em/NotIsolated/General");
  hL1EmNotIsoMult = new TH1D("ElecMult", "Elec Multiplicity", 10, 0, 10);
  hL1EmNotIso1Pt  = new TH1D("Elec1Pt",  "Elec 1 Pt ",        100, 0, 100);
  hL1EmNotIso2Pt  = new TH1D("Elec2Pt",  "Elec 2 Pt ",        100, 0, 100);
  hL1EmNotIso1Eta  = new TH1D("Elec1Eta",  "Elec 1 Eta ",        100, -3, 3);
  hL1EmNotIso2Eta  = new TH1D("Elec2Eta",  "Elec 2 Eta ",        100, -3, 3);
  hL1EmNotIso1Phi  = new TH1D("Elec1Phi",  "Elec 1 Phi ",        100, -3.2, 3.2);
  hL1EmNotIso2Phi  = new TH1D("Elec2Phi",  "Elec 2 Phi ",        100, -3.2, 3.2);
  
  gDirectory->cd("/L1Em/NotIsolated/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "ElecMult_" + (*l1Names_)[i];
    myHistoTitle = "Elec Multiplicity for L1 path " + (*l1Names_)[i];
    hL1EmNotIsoMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Elec1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Pt for L1 path " + (*l1Names_)[i];
    hL1EmNotIso1PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Pt for L1 path " + (*l1Names_)[i];
    hL1EmNotIso2PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Eta for L1 path " + (*l1Names_)[i];
    hL1EmNotIso1EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Eta for L1 path " + (*l1Names_)[i];
    hL1EmNotIso2EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Phi for L1 path " + (*l1Names_)[i];
    hL1EmNotIso1PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Elec2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Phi for L1 path " + (*l1Names_)[i];
    hL1EmNotIso2PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
   }

  gDirectory->cd("/L1Em/NotIsolated/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "ElecMult_" + (*hlNames_)[i];
    myHistoTitle = "Elec Multiplicity for HLT path " + (*hlNames_)[i];    
    hL1EmNotIsoMultAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Elec1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Pt for HLT path " + (*hlNames_)[i];
    hL1EmNotIso1PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Pt for HLT path " + (*hlNames_)[i];
    hL1EmNotIso2PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Eta for HLT path " + (*hlNames_)[i];
    hL1EmNotIso1EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Eta for HLT path " + (*hlNames_)[i];
    hL1EmNotIso2EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Phi for HLT path " + (*hlNames_)[i];
    hL1EmNotIso1PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Elec2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Phi for HLT path " + (*hlNames_)[i];
    hL1EmNotIso2PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }
  gDirectory->cd();

  //******************
  //Book Muons
  //******************
  
  gDirectory->cd("/L1Muons/General");
  hL1MuonMult = new TH1D("MuonMult", "Muon Multiplicity", 10, 0, 10);
  hL1Muon1Pt  = new TH1D("Muon1Pt",  "Muon 1 Pt ",        100, 0, 100);
  hL1Muon2Pt  = new TH1D("Muon2Pt",  "Muon 2 Pt ",        100, 0, 100);
  hL1Muon1Eta  = new TH1D("Muon1Eta",  "Muon 1 Eta ",        100, -3, 3);
  hL1Muon2Eta  = new TH1D("Muon2Eta",  "Muon 2 Eta ",        100, -3, 3);
  hL1Muon1Phi  = new TH1D("Muon1Phi",  "Muon 1 Phi ",        100, -3.2, 3.2);
  hL1Muon2Phi  = new TH1D("Muon2Phi",  "Muon 2 Phi ",        100, -3.2, 3.2);
  
  gDirectory->cd("/L1Muons/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "MuonMult_" + (*l1Names_)[i];
    myHistoTitle = "Muon Multiplicity for L1 path " + (*l1Names_)[i];
    hL1MuonMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Muon1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Muon 1 Pt for L1 path " + (*l1Names_)[i];
    hL1Muon1PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Muon 2 Pt for L1 path " + (*l1Names_)[i];
    hL1Muon2PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Muon 1 Eta for L1 path " + (*l1Names_)[i];
    hL1Muon1EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Muon2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Muon 2 Eta for L1 path " + (*l1Names_)[i];
    hL1Muon2EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Muon1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Muon 1 Phi for L1 path " + (*l1Names_)[i];
    hL1Muon1PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Muon2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Muon 2 Phi for L1 path " + (*l1Names_)[i];
    hL1Muon2PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  gDirectory->cd("/L1Muons/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "MuonMult_" + (*hlNames_)[i];
    myHistoTitle = "Muon Multiplicity for HLT path " + (*hlNames_)[i];    
    hL1MuonMultAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Muon1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Muon 1 Pt for HLT path " + (*hlNames_)[i];
    hL1Muon1PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Muon 2 Pt for HLT path " + (*hlNames_)[i];
    hL1Muon2PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Muon 1 Eta for HLT path " + (*hlNames_)[i];
    hL1Muon1EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Muon2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Muon 2 Eta for HLT path " + (*hlNames_)[i];
    hL1Muon2EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Muon1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Muon 1 Phi for HLT path " + (*hlNames_)[i];
    hL1Muon1PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Muon2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Muon 2 Phi for HLT path " + (*hlNames_)[i];
    hL1Muon2PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }
  gDirectory->cd();



  //******************
  //Book MET
  //******************
  
  gDirectory->cd("/L1MET/General");
  hL1MET = new TH1D("MET", "MET", 35, 0, 1050);
  hL1METx   = new TH1D("METx", "METx", 35, 0, 1050);
  hL1METy   = new TH1D("METy", "METy", 35, 0, 1050);
  hL1METphi = new TH1D("METphi", "METphi", 100, -3.2, 3.2);
  hL1SumEt  = new TH1D("SumEt", "SumEt", 35, 0, 1050);
  hL1METSignificance = new TH1D("METSignificance", "METSignificance", 100, 0, 100);


  gDirectory->cd("/L1MET/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "MET_" + (*l1Names_)[i];
    myHistoTitle = "MET for L1 path " + (*l1Names_)[i];
    hL1METAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METx_" + (*l1Names_)[i];
    myHistoTitle = "METx for L1 path " + (*l1Names_)[i];
    hL1METxAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METy_" + (*l1Names_)[i];
    myHistoTitle = "METy for L1 path " + (*l1Names_)[i];
    hL1METyAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METphi_" + (*l1Names_)[i];
    myHistoTitle = "METphi for L1 path " + (*l1Names_)[i];
    hL1METphiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "SumEt_" + (*l1Names_)[i];
    myHistoTitle = "SumEt for L1 path " + (*l1Names_)[i];
    hL1SumEtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METSignificance_" + (*l1Names_)[i];
    myHistoTitle = "METSignificance for L1 path " + (*l1Names_)[i];
    hL1METSignificanceAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
  }

  gDirectory->cd("/L1MET/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "MET_" + (*hlNames_)[i];
    myHistoTitle = "MET for HLT path " + (*hlNames_)[i];    
    hL1METAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METx_" + (*hlNames_)[i];
    myHistoTitle = "METx for HLT path " + (*hlNames_)[i];    
    hL1METxAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METy_" + (*hlNames_)[i];
    myHistoTitle = "METy for HLT path " + (*hlNames_)[i];    
    hL1METyAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METphi_" + (*hlNames_)[i];
    myHistoTitle = "METphi for HLT path " + (*hlNames_)[i];    
    hL1METphiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "SumEt_" + (*hlNames_)[i];
    myHistoTitle = "SumEt for HLT path " + (*hlNames_)[i];    
    hL1SumEtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METSignificance_" + (*hlNames_)[i];
    myHistoTitle = "METSignificance for HLT path " + (*hlNames_)[i];    
    hL1METSignificanceAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
  }
  gDirectory->cd();






  //******************
  //Book histos Reco Objects
  //******************

  //******************
  //Book Jets
  //******************
  
  gDirectory->cd("/RecoJets/General");
  hJetMult = new TH1D("JetMult", "Jet Multiplicity", 10, 0, 10);
  hJet1Pt  = new TH1D("Jet1Pt",  "Jet 1 Pt ",        100, 0, 1000);
  hJet2Pt  = new TH1D("Jet2Pt",  "Jet 2 Pt ",        100, 0, 1000);
  hJet1Eta  = new TH1D("Jet1Eta",  "Jet 1 Eta ",        100, -3   , 3   );
  hJet2Eta  = new TH1D("Jet2Eta",  "Jet 2 Eta ",        100, -3   , 3   );
  hJet1Phi  = new TH1D("Jet1Phi",  "Jet 1 Phi ",        100, -3.2 , 3.2 );
  hJet2Phi  = new TH1D("Jet2Phi",  "Jet 2 Phi ",        100, -3.2 , 3.2 );
  
  hDiJetInvMass = new TH1D("DiJetInvMass", "DiJet Invariant Mass", 1000, 0, 1000);
  

  gDirectory->cd("/RecoJets/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "JetMult_" + (*l1Names_)[i];
    myHistoTitle = "Jet Multiplicity for L1 path " + (*l1Names_)[i];
    hJetMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Pt for L1 path " + (*l1Names_)[i];
    hJet1PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Pt for L1 path " + (*l1Names_)[i];
    hJet2PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Eta for L1 path " + (*l1Names_)[i];
    hJet1EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Eta for L1 path " + (*l1Names_)[i];
    hJet2EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Phi for L1 path " + (*l1Names_)[i];
    hJet1PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Phi for L1 path " + (*l1Names_)[i];
    hJet2PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    
    myHistoName = "DiJetInvMass_" + (*l1Names_)[i];
    myHistoTitle = "DiJet Invariant Mass for L1 path " + (*l1Names_)[i];
    hDiJetInvMassAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 1000, 0, 1000));

  }

  gDirectory->cd("/RecoJets/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "JetMult_" + (*hlNames_)[i];
    myHistoTitle = "Jet Multiplicity for HLT path " + (*hlNames_)[i];    
    hJetMultAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Pt for HLT path " + (*hlNames_)[i];
    hJet1PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Pt for HLT path " + (*hlNames_)[i];
    hJet2PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Eta for HLT path " + (*hlNames_)[i];
    hJet1EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Eta for HLT path " + (*hlNames_)[i];
    hJet2EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Phi for HLT path " + (*hlNames_)[i];
    hJet1PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Phi for HLT path " + (*hlNames_)[i];
    hJet2PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));

    myHistoName = "DiJetInvMass_" + (*hlNames_)[i];
    myHistoTitle = "DiJet Invariant Mass for HLT path " + (*hlNames_)[i];
    hDiJetInvMassAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 1000, 0, 1000));

  }
  gDirectory->cd();




  //******************
  //Book Electrons
  //******************
  
  gDirectory->cd("/RecoElectrons/General");
  hElecMult = new TH1D("ElecMult", "Elec Multiplicity", 10, 0, 10);
  hElec1Pt  = new TH1D("Elec1Pt",  "Elec 1 Pt ",        100, 0, 100);
  hElec2Pt  = new TH1D("Elec2Pt",  "Elec 2 Pt ",        100, 0, 100);
  hElec1Eta  = new TH1D("Elec1Eta",  "Elec 1 Eta ",        100, -3, 3);
  hElec2Eta  = new TH1D("Elec2Eta",  "Elec 2 Eta ",        100, -3, 3);
  hElec1Phi  = new TH1D("Elec1Phi",  "Elec 1 Phi ",        100, -3.2, 3.2);
  hElec2Phi  = new TH1D("Elec2Phi",  "Elec 2 Phi ",        100, -3.2, 3.2);

  hDiElecInvMass = new TH1D("DiElecInvMass", "DiElec Invariant Mass", 1000, 0, 1000);
  
  gDirectory->cd("/RecoElectrons/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "ElecMult_" + (*l1Names_)[i];
    myHistoTitle = "Elec Multiplicity for L1 path " + (*l1Names_)[i];
    hElecMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Elec1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Pt for L1 path " + (*l1Names_)[i];
    hElec1PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Pt for L1 path " + (*l1Names_)[i];
    hElec2PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Eta for L1 path " + (*l1Names_)[i];
    hElec1EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Eta for L1 path " + (*l1Names_)[i];
    hElec2EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Phi for L1 path " + (*l1Names_)[i];
    hElec1PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Elec2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Phi for L1 path " + (*l1Names_)[i];
    hElec2PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));

    myHistoName = "DiElecInvMass_" + (*l1Names_)[i];
    myHistoTitle = "DiElec Invariant Mass for L1 path " + (*l1Names_)[i];
    hDiElecInvMassAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 1000, 0, 1000));

  }

  gDirectory->cd("/RecoElectrons/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "ElecMult_" + (*hlNames_)[i];
    myHistoTitle = "Elec Multiplicity for HLT path " + (*hlNames_)[i];    
    hElecMultAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Elec1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Pt for HLT path " + (*hlNames_)[i];
    hElec1PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Pt for HLT path " + (*hlNames_)[i];
    hElec2PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Eta for HLT path " + (*hlNames_)[i];
    hElec1EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Eta for HLT path " + (*hlNames_)[i];
    hElec2EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Phi for HLT path " + (*hlNames_)[i];
    hElec1PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Elec2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Phi for HLT path " + (*hlNames_)[i];
    hElec2PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));

    myHistoName = "DiElecInvMass_" + (*hlNames_)[i];
    myHistoTitle = "DiElec Invariant Mass for HLT path " + (*hlNames_)[i];
    hDiElecInvMassAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 1000, 0, 1000));

  }
  gDirectory->cd();


  //******************
  //Book Muons
  //******************
  
  gDirectory->cd("/RecoMuons/General");
  hMuonMult = new TH1D("MuonMult", "Muon Multiplicity", 10, 0, 10);
  hMuon1Pt  = new TH1D("Muon1Pt",  "Muon 1 Pt ",        100, 0, 100);
  hMuon2Pt  = new TH1D("Muon2Pt",  "Muon 2 Pt ",        100, 0, 100);
  hMuon1Eta  = new TH1D("Muon1Eta",  "Muon 1 Eta ",        100, -3, 3);
  hMuon2Eta  = new TH1D("Muon2Eta",  "Muon 2 Eta ",        100, -3, 3);
  hMuon1Phi  = new TH1D("Muon1Phi",  "Muon 1 Phi ",        100, -3.2, 3.2);
  hMuon2Phi  = new TH1D("Muon2Phi",  "Muon 2 Phi ",        100, -3.2, 3.2);
  
  hDiMuonInvMass = new TH1D("DiMuonInvMass", "DiMuon Invariant Mass", 1000, 0, 1000);

  gDirectory->cd("/RecoMuons/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "MuonMult_" + (*l1Names_)[i];
    myHistoTitle = "Muon Multiplicity for L1 path " + (*l1Names_)[i];
    hMuonMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Muon1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Muon 1 Pt for L1 path " + (*l1Names_)[i];
    hMuon1PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Muon 2 Pt for L1 path " + (*l1Names_)[i];
    hMuon2PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Muon 1 Eta for L1 path " + (*l1Names_)[i];
    hMuon1EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Muon2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Muon 2 Eta for L1 path " + (*l1Names_)[i];
    hMuon2EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Muon1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Muon 1 Phi for L1 path " + (*l1Names_)[i];
    hMuon1PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Muon2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Muon 2 Phi for L1 path " + (*l1Names_)[i];
    hMuon2PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));

    myHistoName = "DiMuonInvMass_" + (*l1Names_)[i];
    myHistoTitle = "DiMuon Invariant Mass for L1 path " + (*l1Names_)[i];
    hDiMuonInvMassAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 1000, 0, 1000));

  }

  gDirectory->cd("/RecoMuons/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "MuonMult_" + (*hlNames_)[i];
    myHistoTitle = "Muon Multiplicity for HLT path " + (*hlNames_)[i];    
    hMuonMultAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Muon1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Muon 1 Pt for HLT path " + (*hlNames_)[i];
    hMuon1PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Muon 2 Pt for HLT path " + (*hlNames_)[i];
    hMuon2PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Muon 1 Eta for HLT path " + (*hlNames_)[i];
    hMuon1EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Muon2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Muon 2 Eta for HLT path " + (*hlNames_)[i];
    hMuon2EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Muon1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Muon 1 Phi for HLT path " + (*hlNames_)[i];
    hMuon1PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Muon2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Muon 2 Phi for HLT path " + (*hlNames_)[i];
    hMuon2PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));

    myHistoName = "DiMuonInvMass_" + (*hlNames_)[i];
    myHistoTitle = "DiMuon Invariant Mass for HLT path " + (*hlNames_)[i];
    hDiMuonInvMassAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 1000, 0, 1000));

  }
  gDirectory->cd();



  //******************
  //Book Photons
  //******************
  
  gDirectory->cd("/RecoPhotons/General");
  hPhotonMult = new TH1D("PhotonMult", "Photon Multiplicity", 10, 0, 10);
  hPhoton1Pt  = new TH1D("Photon1Pt",  "Photon 1 Pt ",        100, 0, 100);
  hPhoton2Pt  = new TH1D("Photon2Pt",  "Photon 2 Pt ",        100, 0, 100);
  hPhoton1Eta  = new TH1D("Photon1Eta",  "Photon 1 Eta ",        100, -3, 3);
  hPhoton2Eta  = new TH1D("Photon2Eta",  "Photon 2 Eta ",        100, -3, 3);
  hPhoton1Phi  = new TH1D("Photon1Phi",  "Photon 1 Phi ",        100, -3.2, 3.2);
  hPhoton2Phi  = new TH1D("Photon2Phi",  "Photon 2 Phi ",        100, -3.2, 3.2);
  
  hDiPhotonInvMass = new TH1D("DiPhotonInvMass", "DiPhoton Invariant Mass", 1000, 0, 1000);

  gDirectory->cd("/RecoPhotons/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "PhotonMult_" + (*l1Names_)[i];
    myHistoTitle = "Photon Multiplicity for L1 path " + (*l1Names_)[i];
    hPhotonMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Photon1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Photon 1 Pt for L1 path " + (*l1Names_)[i];
    hPhoton1PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Photon2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Photon 2 Pt for L1 path " + (*l1Names_)[i];
    hPhoton2PtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Photon1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Photon 1 Eta for L1 path " + (*l1Names_)[i];
    hPhoton1EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Photon2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Photon 2 Eta for L1 path " + (*l1Names_)[i];
    hPhoton2EtaAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Photon1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Photon 1 Phi for L1 path " + (*l1Names_)[i];
    hPhoton1PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Photon2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Photon 2 Phi for L1 path " + (*l1Names_)[i];
    hPhoton2PhiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));

    myHistoName = "DiPhotonInvMass_" + (*l1Names_)[i];
    myHistoTitle = "DiPhoton Invariant Mass for L1 path " + (*l1Names_)[i];
    hDiPhotonInvMassAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 1000, 0, 1000));

  }

  gDirectory->cd("/RecoPhotons/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "PhotonMult_" + (*hlNames_)[i];
    myHistoTitle = "Photon Multiplicity for HLT path " + (*hlNames_)[i];    
    hPhotonMultAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Photon1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Photon 1 Pt for HLT path " + (*hlNames_)[i];
    hPhoton1PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Photon2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Photon 2 Pt for HLT path " + (*hlNames_)[i];
    hPhoton2PtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Photon1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Photon 1 Eta for HLT path " + (*hlNames_)[i];
    hPhoton1EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Photon2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Photon 2 Eta for HLT path " + (*hlNames_)[i];
    hPhoton2EtaAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Photon1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Photon 1 Phi for HLT path " + (*hlNames_)[i];
    hPhoton1PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Photon2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Photon 2 Phi for HLT path " + (*hlNames_)[i];
    hPhoton2PhiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));

    myHistoName = "DiPhotonInvMass_" + (*hlNames_)[i];
    myHistoTitle = "DiPhoton Invariant Mass for HLT path " + (*hlNames_)[i];
    hDiPhotonInvMassAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 1000, 0, 1000));

  }
  gDirectory->cd();



  //******************
  //Book MET
  //******************
  
  gDirectory->cd("/RecoMET/General");
  hMET = new TH1D("MET", "MET", 35, 0, 1050);
  hMETx   = new TH1D("METx", "METx", 35, 0, 1050);
  hMETy   = new TH1D("METy", "METy", 35, 0, 1050);
  hMETphi = new TH1D("METphi", "METphi", 100, -3.2, 3.2);
  hSumEt  = new TH1D("SumEt" , "SumEt",  35, 0, 1050);
  hMETSignificance = new TH1D("METSignificance", "METSignificance", 100, 0, 100);
  gDirectory->cd("/RecoMET/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "MET_" + (*l1Names_)[i];
    myHistoTitle = "MET for L1 path " + (*l1Names_)[i];
    hMETAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METx_" + (*l1Names_)[i];
    myHistoTitle = "METx for L1 path " + (*l1Names_)[i];
    hMETxAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METy_" + (*l1Names_)[i];
    myHistoTitle = "METy for L1 path " + (*l1Names_)[i];
    hMETyAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METPhi_" + (*l1Names_)[i];
    myHistoTitle = "METPhi for L1 path " + (*l1Names_)[i];
    hMETphiAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "SumEt_" + (*l1Names_)[i];
    myHistoTitle = "SumEt for L1 path " + (*l1Names_)[i];
    hSumEtAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METSignificance_" + (*l1Names_)[i];
    myHistoTitle = "METSignificance for L1 path " + (*l1Names_)[i];
    hMETSignificanceAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
  }

  gDirectory->cd("/RecoMET/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "MET_" + (*hlNames_)[i];
    myHistoTitle = "MET for HLT path " + (*hlNames_)[i];    
    hMETAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METx_" + (*hlNames_)[i];
    myHistoTitle = "METx for HLT path " + (*hlNames_)[i];    
    hMETxAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METy_" + (*hlNames_)[i];
    myHistoTitle = "METy for HLT path " + (*hlNames_)[i];    
    hMETyAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METPhi_" + (*hlNames_)[i];
    myHistoTitle = "METPhi for HLT path " + (*hlNames_)[i];    
    hMETphiAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2 ));
    myHistoName = "SumEt_" + (*hlNames_)[i];
    myHistoTitle = "SumEt for HLT path " + (*hlNames_)[i];    
    hSumEtAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METSignificance_" + (*hlNames_)[i];
    myHistoTitle = "METSignificance for HLT path " + (*hlNames_)[i];    
    hMETSignificanceAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
  }
  gDirectory->cd();




}


void PlotMaker::writeHistos() {
  
  //******************
  //Write histos for L1 Objects
  //******************
  
  
  //******************
  //Write Jets
  //******************

  gDirectory->cd("/L1Jets/Central/General");
  hL1CentralJetMult ->Write();
  hL1CentralJet1Pt  ->Write();
  hL1CentralJet2Pt  ->Write(); 
  hL1CentralJet1Eta ->Write();
  hL1CentralJet2Eta ->Write();
  hL1CentralJet1Phi ->Write();
  hL1CentralJet2Phi ->Write();
  
  gDirectory->cd("/L1Jets/Central/L1");
  for(unsigned int i=0; i<hL1CentralJetMultAfterL1.size(); i++){
    hL1CentralJetMultAfterL1[i]->Write();
    hL1CentralJet1PtAfterL1[i] ->Write();
    hL1CentralJet2PtAfterL1[i] ->Write();
    hL1CentralJet1EtaAfterL1[i]->Write();
    hL1CentralJet2EtaAfterL1[i]->Write();
    hL1CentralJet1PhiAfterL1[i]->Write();
    hL1CentralJet2PhiAfterL1[i]->Write();
  }

  gDirectory->cd("/L1Jets/Central/HLT");
  for(unsigned int i=0; i<hL1CentralJetMultAfterHLT.size(); i++){
    hL1CentralJetMultAfterHLT[i]->Write();
    hL1CentralJet1PtAfterHLT[i]->Write();
    hL1CentralJet2PtAfterHLT[i]->Write();
    hL1CentralJet1EtaAfterHLT[i]->Write();
    hL1CentralJet2EtaAfterHLT[i]->Write();
    hL1CentralJet1PhiAfterHLT[i]->Write();
    hL1CentralJet2PhiAfterHLT[i]->Write();
  }

  gDirectory->cd("/L1Jets/Forward/General");
  hL1ForwardJetMult ->Write();
  hL1ForwardJet1Pt  ->Write();
  hL1ForwardJet2Pt  ->Write();
  hL1ForwardJet1Eta ->Write();
  hL1ForwardJet2Eta ->Write();
  hL1ForwardJet1Phi ->Write();
  hL1ForwardJet2Phi ->Write();

  gDirectory->cd("/L1Jets/Forward/L1");
  for(unsigned int i=0; i<hL1ForwardJetMultAfterL1.size(); i++){
    hL1ForwardJetMultAfterL1[i]->Write();
    hL1ForwardJet1PtAfterL1[i]->Write();
    hL1ForwardJet2PtAfterL1[i]->Write();
    hL1ForwardJet1EtaAfterL1[i]->Write();
    hL1ForwardJet2EtaAfterL1[i]->Write();
    hL1ForwardJet1PhiAfterL1[i]->Write();
    hL1ForwardJet2PhiAfterL1[i]->Write();
  }

  gDirectory->cd("/L1Jets/Forward/HLT");
  for(unsigned int i=0; i<hL1ForwardJetMultAfterHLT.size(); i++){
    hL1ForwardJetMultAfterHLT[i]->Write();
    hL1ForwardJet1PtAfterHLT[i]->Write();
    hL1ForwardJet2PtAfterHLT[i]->Write();
    hL1ForwardJet1EtaAfterHLT[i]->Write();
    hL1ForwardJet2EtaAfterHLT[i]->Write();
    hL1ForwardJet1PhiAfterHLT[i]->Write();
    hL1ForwardJet2PhiAfterHLT[i]->Write();
  }

  gDirectory->cd("/L1Jets/Tau/General");
  hL1TauJetMult  ->Write();
  hL1TauJet1Pt   ->Write();
  hL1TauJet2Pt   ->Write();
  hL1TauJet1Eta  ->Write();
  hL1TauJet2Eta  ->Write();
  hL1TauJet1Phi  ->Write();
  hL1TauJet2Phi  ->Write();

  gDirectory->cd("/L1Jets/Tau/L1");
  for(unsigned int i=0; i<hL1TauJetMultAfterL1.size(); i++){
    hL1TauJetMultAfterL1[i]->Write();
    hL1TauJet1PtAfterL1[i]->Write();
    hL1TauJet2PtAfterL1[i]->Write();
    hL1TauJet1EtaAfterL1[i]->Write();
    hL1TauJet2EtaAfterL1[i]->Write();
    hL1TauJet1PhiAfterL1[i]->Write();
    hL1TauJet2PhiAfterL1[i]->Write();
  }

  gDirectory->cd("/L1Jets/Tau/HLT");
  for(unsigned int i=0; i<hL1TauJetMultAfterHLT.size(); i++){
    hL1TauJetMultAfterHLT[i]->Write();
    hL1TauJet1PtAfterHLT[i]->Write();
    hL1TauJet2PtAfterHLT[i]->Write();
    hL1TauJet1EtaAfterHLT[i]->Write();
    hL1TauJet2EtaAfterHLT[i]->Write();
    hL1TauJet1PhiAfterHLT[i]->Write();
    hL1TauJet2PhiAfterHLT[i]->Write();
  }



  gDirectory->cd("/L1Em/Isolated/General");
  hL1EmIsoMult  ->Write();
  hL1EmIso1Pt   ->Write();
  hL1EmIso2Pt   ->Write();
  hL1EmIso1Eta  ->Write();
  hL1EmIso2Eta  ->Write();
  hL1EmIso1Phi  ->Write();
  hL1EmIso2Phi  ->Write();
  
  gDirectory->cd("/L1Em/Isolated/L1");
  for(unsigned int i=0; i<hL1EmIsoMultAfterL1.size(); i++){
    hL1EmIsoMultAfterL1[i]->Write();
    hL1EmIso1PtAfterL1[i]->Write();
    hL1EmIso2PtAfterL1[i]->Write();
    hL1EmIso1EtaAfterL1[i]->Write();
    hL1EmIso2EtaAfterL1[i]->Write();
    hL1EmIso1PhiAfterL1[i]->Write();
    hL1EmIso2PhiAfterL1[i]->Write();
  }

  gDirectory->cd("/L1Em/Isolated/HLT");
  for(unsigned int i=0; i<hL1EmIsoMultAfterHLT.size(); i++){
    hL1EmIsoMultAfterHLT[i]->Write();
    hL1EmIso1PtAfterHLT[i]->Write();
    hL1EmIso2PtAfterHLT[i]->Write();
    hL1EmIso1EtaAfterHLT[i]->Write();
    hL1EmIso2EtaAfterHLT[i]->Write();
    hL1EmIso1PhiAfterHLT[i]->Write();
    hL1EmIso2PhiAfterHLT[i]->Write();
  }
  gDirectory->cd();



  gDirectory->cd("/L1Em/NotIsolated/General");
  hL1EmNotIsoMult ->Write(); 
  hL1EmNotIso1Pt  ->Write(); 
  hL1EmNotIso2Pt  ->Write(); 
  hL1EmNotIso1Eta ->Write(); 
  hL1EmNotIso2Eta ->Write(); 
  hL1EmNotIso1Phi ->Write(); 
  hL1EmNotIso2Phi ->Write(); 
  
  gDirectory->cd("/L1Em/NotIsolated/L1");
  for(unsigned int i=0; i<hL1EmNotIsoMultAfterL1.size(); i++){
    hL1EmNotIsoMultAfterL1[i]->Write();
    hL1EmNotIso1PtAfterL1[i]->Write();
    hL1EmNotIso2PtAfterL1[i]->Write();
    hL1EmNotIso1EtaAfterL1[i]->Write();
    hL1EmNotIso2EtaAfterL1[i]->Write();
    hL1EmNotIso1PhiAfterL1[i]->Write();
    hL1EmNotIso2PhiAfterL1[i]->Write();
   }

  gDirectory->cd("/L1Em/NotIsolated/HLT");
  for(unsigned int i=0; i<hL1EmNotIsoMultAfterHLT.size(); i++){
    hL1EmNotIsoMultAfterHLT[i]->Write();
    hL1EmNotIso1PtAfterHLT[i]->Write();
    hL1EmNotIso2PtAfterHLT[i]->Write();
    hL1EmNotIso1EtaAfterHLT[i]->Write();
    hL1EmNotIso2EtaAfterHLT[i]->Write();
    hL1EmNotIso1PhiAfterHLT[i]->Write();
    hL1EmNotIso2PhiAfterHLT[i]->Write();
  }
  gDirectory->cd();

  //******************
  //Book Muons
  //******************
  
  gDirectory->cd("/L1Muons/General");
  hL1MuonMult  ->Write();
  hL1Muon1Pt   ->Write();
  hL1Muon2Pt   ->Write();
  hL1Muon1Eta  ->Write();
  hL1Muon2Eta  ->Write();
  hL1Muon1Phi  ->Write();
  hL1Muon2Phi  ->Write();
  
  gDirectory->cd("/L1Muons/L1");
  for(unsigned int i=0; i<hL1MuonMultAfterL1.size(); i++){
    hL1MuonMultAfterL1[i]->Write();
    hL1Muon1PtAfterL1[i]->Write();
    hL1Muon2PtAfterL1[i]->Write();
    hL1Muon1EtaAfterL1[i]->Write();
    hL1Muon2EtaAfterL1[i]->Write();
    hL1Muon1PhiAfterL1[i]->Write();
    hL1Muon2PhiAfterL1[i]->Write();
  }

  gDirectory->cd("/L1Muons/HLT");
  for(unsigned int i=0; i<hL1MuonMultAfterHLT.size(); i++){
    hL1MuonMultAfterHLT[i]->Write();
    hL1Muon1PtAfterHLT[i]->Write();
    hL1Muon2PtAfterHLT[i]->Write();
    hL1Muon1EtaAfterHLT[i]->Write();
    hL1Muon2EtaAfterHLT[i]->Write();
    hL1Muon1PhiAfterHLT[i]->Write();
    hL1Muon2PhiAfterHLT[i]->Write();
  }
  gDirectory->cd();



  //******************
  //Book MET
  //******************
  
  gDirectory->cd("/L1MET/General");
  hL1MET    	     ->Write();
  hL1METx   	     ->Write();
  hL1METy   	     ->Write();
  hL1METphi 	     ->Write();
  hL1SumEt  	     ->Write();
  hL1METSignificance ->Write();


  gDirectory->cd("/L1MET/L1");
  for(unsigned int i=0; i<hL1METAfterL1.size(); i++){
    hL1METAfterL1[i]->Write();
    hL1METxAfterL1[i]->Write();
    hL1METyAfterL1[i]->Write();
    hL1METphiAfterL1[i]->Write();
    hL1SumEtAfterL1[i]->Write();
    hL1METSignificanceAfterL1[i]->Write();
  }

  gDirectory->cd("/L1MET/HLT");
  for(unsigned int i=0; i<hL1METAfterHLT.size(); i++){
    hL1METAfterHLT[i]->Write();
    hL1METxAfterHLT[i]->Write();
    hL1METyAfterHLT[i]->Write();
    hL1METphiAfterHLT[i]->Write();
    hL1SumEtAfterHLT[i]->Write();
    hL1METSignificanceAfterHLT[i]->Write();
  }
  gDirectory->cd();



  //******************
  //Write histos Reco Objects
  //******************

  //******************
  //Write Jets
  //******************
  
  gDirectory->cd("/RecoJets/General");
  hJetMult  ->Write();
  hJet1Pt   ->Write();
  hJet2Pt   ->Write();
  hJet1Eta  ->Write();
  hJet2Eta  ->Write();
  hJet1Phi  ->Write();
  hJet2Phi  ->Write();
  
  hDiJetInvMass ->Write();
  

  gDirectory->cd("/RecoJets/L1");
  for(unsigned int i=0; i<hJetMultAfterL1.size(); i++){
    hJetMultAfterL1[i]->Write();
    hJet1PtAfterL1[i]->Write();
    hJet2PtAfterL1[i]->Write();
    hJet1EtaAfterL1[i]->Write();
    hJet2EtaAfterL1[i]->Write();
    hJet1PhiAfterL1[i]->Write();
    hJet2PhiAfterL1[i]->Write();
    
    hDiJetInvMassAfterL1[i]->Write();

  }

  gDirectory->cd("/RecoJets/HLT");
  for(unsigned int i=0; i<hJetMultAfterHLT.size(); i++){
    hJetMultAfterHLT[i]->Write();
    hJet1PtAfterHLT[i]->Write();
    hJet2PtAfterHLT[i]->Write();
    hJet1EtaAfterHLT[i]->Write();
    hJet2EtaAfterHLT[i]->Write();
    hJet1PhiAfterHLT[i]->Write();
    hJet2PhiAfterHLT[i]->Write();

    hDiJetInvMassAfterHLT[i]->Write();

  }
  gDirectory->cd();




  //******************
  //Book Electrons
  //******************
  
  gDirectory->cd("/RecoElectrons/General");
  hElecMult  ->Write();
  hElec1Pt   ->Write();
  hElec2Pt   ->Write();
  hElec1Eta  ->Write();
  hElec2Eta  ->Write();
  hElec1Phi  ->Write();
  hElec2Phi  ->Write();

  hDiElecInvMass ->Write();
  
  gDirectory->cd("/RecoElectrons/L1");
  for(unsigned int i=0; i<hElecMultAfterL1.size(); i++){
    hElecMultAfterL1[i]->Write();
    hElec1PtAfterL1[i]->Write();
    hElec2PtAfterL1[i]->Write();
    hElec1EtaAfterL1[i]->Write();
    hElec2EtaAfterL1[i]->Write();
    hElec1PhiAfterL1[i]->Write();
    hElec2PhiAfterL1[i]->Write();

    hDiElecInvMassAfterL1[i]->Write();

  }

  gDirectory->cd("/RecoElectrons/HLT");
  for(unsigned int i=0; i<hElecMultAfterHLT.size(); i++){
    hElecMultAfterHLT[i]->Write();
    hElec1PtAfterHLT[i]->Write();
    hElec2PtAfterHLT[i]->Write();
    hElec1EtaAfterHLT[i]->Write();
    hElec2EtaAfterHLT[i]->Write();
    hElec1PhiAfterHLT[i]->Write();
    hElec2PhiAfterHLT[i]->Write();

    hDiElecInvMassAfterHLT[i]->Write();

  }
  gDirectory->cd();


  //******************
  //Book Muons
  //******************
  
  gDirectory->cd("/RecoMuons/General");
  hMuonMult  ->Write();
  hMuon1Pt   ->Write();
  hMuon2Pt   ->Write();
  hMuon1Eta  ->Write();
  hMuon2Eta  ->Write();
  hMuon1Phi  ->Write();
  hMuon2Phi  ->Write();
  
  hDiMuonInvMass ->Write();

  gDirectory->cd("/RecoMuons/L1");
  for(unsigned int i=0; i<hMuonMultAfterL1.size(); i++){
    hMuonMultAfterL1[i]->Write();
    hMuon1PtAfterL1[i]->Write();
    hMuon2PtAfterL1[i]->Write();
    hMuon1EtaAfterL1[i]->Write();
    hMuon2EtaAfterL1[i]->Write();
    hMuon1PhiAfterL1[i]->Write();
    hMuon2PhiAfterL1[i]->Write();

    hDiMuonInvMassAfterL1[i]->Write();

  }

  gDirectory->cd("/RecoMuons/HLT");
  for(unsigned int i=0; i<hMuonMultAfterHLT.size(); i++){
    hMuonMultAfterHLT[i]->Write();
    hMuon1PtAfterHLT[i]->Write();
    hMuon2PtAfterHLT[i]->Write();
    hMuon1EtaAfterHLT[i]->Write();
    hMuon2EtaAfterHLT[i]->Write();
    hMuon1PhiAfterHLT[i]->Write();
    hMuon2PhiAfterHLT[i]->Write();

    hDiMuonInvMassAfterHLT[i]->Write();

  }
  gDirectory->cd();



  //******************
  //Book Photons
  //******************
  
  gDirectory->cd("/RecoPhotons/General");
  hPhotonMult  ->Write();
  hPhoton1Pt   ->Write();
  hPhoton2Pt   ->Write();
  hPhoton1Eta  ->Write();
  hPhoton2Eta  ->Write();
  hPhoton1Phi  ->Write();
  hPhoton2Phi  ->Write();
  
  hDiPhotonInvMass ->Write();

  gDirectory->cd("/RecoPhotons/L1");
  for(unsigned int i=0; i<hPhotonMultAfterL1.size(); i++){
    hPhotonMultAfterL1[i]->Write();
    hPhoton1PtAfterL1[i]->Write();
    hPhoton2PtAfterL1[i]->Write();
    hPhoton1EtaAfterL1[i]->Write();
    hPhoton2EtaAfterL1[i]->Write();
    hPhoton1PhiAfterL1[i]->Write();
    hPhoton2PhiAfterL1[i]->Write();

    hDiPhotonInvMassAfterL1[i]->Write();

  }

  gDirectory->cd("/RecoPhotons/HLT");
  for(unsigned int i=0; i<hPhotonMultAfterHLT.size(); i++){
    hPhotonMultAfterHLT[i]->Write();
    hPhoton1PtAfterHLT[i]->Write();
    hPhoton2PtAfterHLT[i]->Write();
    hPhoton1EtaAfterHLT[i]->Write();
    hPhoton2EtaAfterHLT[i]->Write();
    hPhoton1PhiAfterHLT[i]->Write();
    hPhoton2PhiAfterHLT[i]->Write();

    hDiPhotonInvMassAfterHLT[i]->Write();

  }
  gDirectory->cd();



  //******************
  //Book MET
  //******************
  
  gDirectory->cd("/RecoMET/General");
  hMET    	   ->Write();
  hMETx   	   ->Write();
  hMETy   	   ->Write();
  hMETphi 	   ->Write();
  hSumEt  	   ->Write();
  hMETSignificance ->Write();

  gDirectory->cd("/RecoMET/L1");
  for(unsigned int i=0; i<hMETAfterL1.size(); i++){
    hMETAfterL1[i]->Write();
    hMETxAfterL1[i]->Write();
    hMETyAfterL1[i]->Write();
    hMETphiAfterL1[i]->Write();
    hSumEtAfterL1[i]->Write();
    hMETSignificanceAfterL1[i]->Write();
  }

  gDirectory->cd("/RecoMET/HLT");
  for(unsigned int i=0; i<hMETAfterHLT.size(); i++){
    hMETAfterHLT[i]->Write();
    hMETxAfterHLT[i]->Write();
    hMETyAfterHLT[i]->Write();
    hMETphiAfterHLT[i]->Write();
    hSumEtAfterHLT[i]->Write();
    hMETSignificanceAfterHLT[i]->Write();
  }
  gDirectory->cd();






}






void PlotMaker::handleObjects(const edm::Event& iEvent)
{


  //**************************************************
  // Get the L1 Objects through the l1extra Collection
  //**************************************************

  //Get the EM objects

  Handle<l1extra::L1EmParticleCollection> theL1EmIsoHandle, theL1EmNotIsoHandle;
  iEvent.getByLabel(m_l1extra,"Isolated",theL1EmIsoHandle);
  iEvent.getByLabel(m_l1extra,"NonIsolated",theL1EmNotIsoHandle);
  theL1EmIsoCollection = *theL1EmIsoHandle;
  std::sort(theL1EmIsoCollection.begin(), theL1EmIsoCollection.end(), PtSorter());
  theL1EmNotIsoCollection = *theL1EmNotIsoHandle;
  std::sort(theL1EmNotIsoCollection.begin(), theL1EmNotIsoCollection.end(), PtSorter());

  //Get the Muons  
  Handle<l1extra::L1MuonParticleCollection> theL1MuonHandle;
  iEvent.getByLabel(m_l1extra,theL1MuonHandle);
  theL1MuonCollection = *theL1MuonHandle;
  std::sort(theL1MuonCollection.begin(), theL1MuonCollection.end(),PtSorter());

  //Get the Jets
  Handle<l1extra::L1JetParticleCollection> theL1CentralJetHandle,theL1ForwardJetHandle,theL1TauJetHandle;
  iEvent.getByLabel(m_l1extra,"Central",theL1CentralJetHandle);
  iEvent.getByLabel(m_l1extra,"Forward",theL1ForwardJetHandle);
  iEvent.getByLabel(m_l1extra,"Tau",theL1TauJetHandle);
  theL1CentralJetCollection = *theL1CentralJetHandle;
  std::sort(theL1CentralJetCollection.begin(), theL1CentralJetCollection.end(), PtSorter());
  theL1ForwardJetCollection = *theL1ForwardJetHandle;
  std::sort(theL1ForwardJetCollection.begin(), theL1ForwardJetCollection.end(), PtSorter());
  theL1TauJetCollection = *theL1TauJetHandle;
  std::sort(theL1TauJetCollection.begin(), theL1TauJetCollection.end(), PtSorter());


  //Get the MET
  Handle<l1extra::L1EtMissParticleCollection> theL1METHandle;
  iEvent.getByLabel(m_l1extra,theL1METHandle);
  theL1METCollection = *theL1METHandle;
  std::sort(theL1METCollection.begin(), theL1METCollection.end(),PtSorter());

  //***********************************************
  // Get the RECO Objects
  //***********************************************


  //Get the electrons
  Handle<PixelMatchGsfElectronCollection> theElectronCollectionHandle; 
  iEvent.getByLabel(m_electronSrc, theElectronCollectionHandle);
  theElectronCollection = *theElectronCollectionHandle;
  std::sort(theElectronCollection.begin(), theElectronCollection.end(), PtSorter());

  //Get the Muons
  Handle<MuonCollection> theMuonCollectionHandle; 
  iEvent.getByLabel(m_muonSrc, theMuonCollectionHandle);
  theMuonCollection = *theMuonCollectionHandle;
  std::sort(theMuonCollection.begin(), theMuonCollection.end(), PtSorter());

  //Get the Photons
  Handle<PhotonCollection> thePhotonCollectionHandle; 
  iEvent.getByLabel(m_photonProducerSrc, m_photonSrc, thePhotonCollectionHandle);
  thePhotonCollection = *thePhotonCollectionHandle;
  std::sort(thePhotonCollection.begin(), thePhotonCollection.end(), PtSorter());

  //Get the CaloJets
  Handle<CaloJetCollection> theCaloJetCollectionHandle;
  iEvent.getByLabel(m_jetsSrc, theCaloJetCollectionHandle);
  theCaloJetCollection = *theCaloJetCollectionHandle;
  std::sort(theCaloJetCollection.begin(), theCaloJetCollection.end(), PtSorter());

  //Get the CaloMET
  Handle<CaloMETCollection> theCaloMETCollectionHandle;
  iEvent.getByLabel(m_calometSrc, theCaloMETCollectionHandle);
  theCaloMETCollection = *theCaloMETCollectionHandle;
}

double PlotMaker::invariantMass(reco::Particle* p1, reco::Particle* p2) {
  double mass = sqrt( (p1->energy() + p2->energy())*(p1->energy() + p2->energy()) -
	       (p1->px() + p2->px())*(p1->px() + p2->px()) -
	       (p1->py() + p2->py())*(p1->py() + p2->py()) -
	       (p1->pz() + p2->pz())*(p1->pz() + p2->pz()) );


//   cout << "p1->energy() = " << p1->energy() << " p2->energy() = " << p2->energy() << endl;
//   cout << "p1->px() = " << p1->px() << " p2->px() = " << p2->px() << endl;
//   cout << "p1->py() = " << p1->py() << " p2->py() = " << p2->py() << endl;
//   cout << "p1->pz() = " << p1->pz() << " p2->pz() = " << p2->pz() << endl;
//   cout << "invmass = " << mass << endl;


  return mass;
}

