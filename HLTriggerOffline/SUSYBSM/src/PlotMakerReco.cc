/*  \class PlotMakerReco
*
*  Class to produce some plots of Off-line variables in the TriggerValidation Code
*
*  Author: Massimiliano Chiorboli      Date: September 2007
//         Maurizio Pierini
//         Maria Spiropulu
*
*/
#include "HLTriggerOffline/SUSYBSM/interface/PlotMakerReco.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TDirectory.h"

#include "HLTriggerOffline/SUSYBSM/interface/PtSorter.h"


using namespace edm;
using namespace reco;
using namespace std;
using namespace l1extra;

PlotMakerReco::PlotMakerReco(edm::ParameterSet PlotMakerRecoInput)
{
  m_electronSrc  	 = PlotMakerRecoInput.getParameter<string>("electrons");
  m_muonSrc    	 	 = PlotMakerRecoInput.getParameter<string>("muons");
  m_jetsSrc    	 	 = PlotMakerRecoInput.getParameter<string>("jets");
  m_photonProducerSrc  	 = PlotMakerRecoInput.getParameter<string>("photonProducer");
  m_photonSrc  	 	 = PlotMakerRecoInput.getParameter<string>("photons");
  m_calometSrc 	 	 = PlotMakerRecoInput.getParameter<string>("calomet");

  def_electronPtMin = PlotMakerRecoInput.getParameter<double>("def_electronPtMin");
  def_muonPtMin     = PlotMakerRecoInput.getParameter<double>("def_muonPtMin");
  def_jetPtMin      = PlotMakerRecoInput.getParameter<double>("def_jetPtMin");
  def_photonPtMin   = PlotMakerRecoInput.getParameter<double>("def_photonPtMin");

  binFactor         = PlotMakerRecoInput.getParameter<int>("BinFactor");

  dirname_          = PlotMakerRecoInput.getParameter<std::string>("dirname");

  edm::LogInfo("PlotMakerRecoObjects") << endl;
  edm::LogInfo("PlotMakerRecoObjects") << "Object definition cuts:" << endl;
  edm::LogInfo("PlotMakerRecoObjects") << " def_electronPtMin  " << def_electronPtMin << endl;
  edm::LogInfo("PlotMakerRecoObjects") << " def_muonPtMin      " << def_muonPtMin     << endl;
  edm::LogInfo("PlotMakerRecoObjects") << " def_jetPtMin       " << def_jetPtMin      << endl;
  edm::LogInfo("PlotMakerRecoObjects") << " def_photonPtMin    " << def_photonPtMin   << endl;


}

void PlotMakerReco::fillPlots(const edm::Event& iEvent)
{
  this->handleObjects(iEvent);



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




void PlotMakerReco::bookHistos(DQMStore * dbe_, std::vector<int>* l1bits, std::vector<int>* hltbits, 
			   std::vector<std::string>* l1Names_, std::vector<std::string>* hlNames_)
{

  this->setBits(l1bits, hltbits);


  //******************
  //Book histos Reco Objects
  //******************

  //******************
  //Book Jets
  //******************
  
  dbe_->setCurrentFolder(dirname_+"/RecoJets/General");
  hJetMult = dbe_->book1D("JetMult", "Jet Multiplicity", 10, 0, 10);
  hJet1Pt  = dbe_->book1D("Jet1Pt",  "Jet 1 Pt ",        100, 0, 1000);
  hJet2Pt  = dbe_->book1D("Jet2Pt",  "Jet 2 Pt ",        100, 0, 1000);
  hJet1Eta  = dbe_->book1D("Jet1Eta",  "Jet 1 Eta ",        10*binFactor, -3   , 3   );
  hJet2Eta  = dbe_->book1D("Jet2Eta",  "Jet 2 Eta ",        10*binFactor, -3   , 3   );
  hJet1Phi  = dbe_->book1D("Jet1Phi",  "Jet 1 Phi ",        10*binFactor, -3.2 , 3.2 );
  hJet2Phi  = dbe_->book1D("Jet2Phi",  "Jet 2 Phi ",        10*binFactor, -3.2 , 3.2 );
  
  hDiJetInvMass = dbe_->book1D("DiJetInvMass", "DiJet Invariant Mass", 100*binFactor, 0, 1000);
  

  dbe_->setCurrentFolder(dirname_+"/RecoJets/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "JetMult_" + (*l1Names_)[i];
    myHistoTitle = "Jet Multiplicity for L1 path " + (*l1Names_)[i];
    hJetMultAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Pt for L1 path " + (*l1Names_)[i];
    hJet1PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Pt for L1 path " + (*l1Names_)[i];
    hJet2PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Eta for L1 path " + (*l1Names_)[i];
    hJet1EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Jet2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Eta for L1 path " + (*l1Names_)[i];
    hJet2EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Jet1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Phi for L1 path " + (*l1Names_)[i];
    hJet1PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Phi for L1 path " + (*l1Names_)[i];
    hJet2PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));
    
    myHistoName = "DiJetInvMass_" + (*l1Names_)[i];
    myHistoTitle = "DiJet Invariant Mass for L1 path " + (*l1Names_)[i];
    hDiJetInvMassAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100*binFactor, 0, 1000));

  }

  dbe_->setCurrentFolder(dirname_+"/RecoJets/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "JetMult_" + (*hlNames_)[i];
    myHistoTitle = "Jet Multiplicity for HLT path " + (*hlNames_)[i];    
    hJetMultAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Pt for HLT path " + (*hlNames_)[i];
    hJet1PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Pt for HLT path " + (*hlNames_)[i];
    hJet2PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Eta for HLT path " + (*hlNames_)[i];
    hJet1EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Jet2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Eta for HLT path " + (*hlNames_)[i];
    hJet2EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Jet1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Phi for HLT path " + (*hlNames_)[i];
    hJet1PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Phi for HLT path " + (*hlNames_)[i];
    hJet2PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));

    myHistoName = "DiJetInvMass_" + (*hlNames_)[i];
    myHistoTitle = "DiJet Invariant Mass for HLT path " + (*hlNames_)[i];
    hDiJetInvMassAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100*binFactor, 0, 1000));

  }
  dbe_->setCurrentFolder(dirname_);




  //******************
  //Book Electrons
  //******************
  
  dbe_->setCurrentFolder(dirname_+"/RecoElectrons/General");
  hElecMult = dbe_->book1D("ElecMult", "Elec Multiplicity", 10, 0, 10);
  hElec1Pt  = dbe_->book1D("Elec1Pt",  "Elec 1 Pt ",        100, 0, 100);
  hElec2Pt  = dbe_->book1D("Elec2Pt",  "Elec 2 Pt ",        100, 0, 100);
  hElec1Eta  = dbe_->book1D("Elec1Eta",  "Elec 1 Eta ",        10*binFactor, -3, 3);
  hElec2Eta  = dbe_->book1D("Elec2Eta",  "Elec 2 Eta ",        10*binFactor, -3, 3);
  hElec1Phi  = dbe_->book1D("Elec1Phi",  "Elec 1 Phi ",        10*binFactor, -3.2, 3.2);
  hElec2Phi  = dbe_->book1D("Elec2Phi",  "Elec 2 Phi ",        10*binFactor, -3.2, 3.2);

  hDiElecInvMass = dbe_->book1D("DiElecInvMass", "DiElec Invariant Mass", 100*binFactor, 0, 1000);
  
  dbe_->setCurrentFolder(dirname_+"/RecoElectrons/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "ElecMult_" + (*l1Names_)[i];
    myHistoTitle = "Elec Multiplicity for L1 path " + (*l1Names_)[i];
    hElecMultAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Elec1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Pt for L1 path " + (*l1Names_)[i];
    hElec1PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Pt for L1 path " + (*l1Names_)[i];
    hElec2PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Eta for L1 path " + (*l1Names_)[i];
    hElec1EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Elec2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Eta for L1 path " + (*l1Names_)[i];
    hElec2EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Elec1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Phi for L1 path " + (*l1Names_)[i];
    hElec1PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));
    myHistoName = "Elec2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Phi for L1 path " + (*l1Names_)[i];
    hElec2PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));

    myHistoName = "DiElecInvMass_" + (*l1Names_)[i];
    myHistoTitle = "DiElec Invariant Mass for L1 path " + (*l1Names_)[i];
    hDiElecInvMassAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100*binFactor, 0, 1000));

  }

  dbe_->setCurrentFolder(dirname_+"/RecoElectrons/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "ElecMult_" + (*hlNames_)[i];
    myHistoTitle = "Elec Multiplicity for HLT path " + (*hlNames_)[i];    
    hElecMultAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Elec1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Pt for HLT path " + (*hlNames_)[i];
    hElec1PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Pt for HLT path " + (*hlNames_)[i];
    hElec2PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Eta for HLT path " + (*hlNames_)[i];
    hElec1EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Elec2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Eta for HLT path " + (*hlNames_)[i];
    hElec2EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Elec1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Phi for HLT path " + (*hlNames_)[i];
    hElec1PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));
    myHistoName = "Elec2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Phi for HLT path " + (*hlNames_)[i];
    hElec2PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));

    myHistoName = "DiElecInvMass_" + (*hlNames_)[i];
    myHistoTitle = "DiElec Invariant Mass for HLT path " + (*hlNames_)[i];
    hDiElecInvMassAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100*binFactor, 0, 1000));

  }
  dbe_->setCurrentFolder(dirname_);


  //******************
  //Book Muons
  //******************
  
  dbe_->setCurrentFolder(dirname_+"/RecoMuons/General");
  hMuonMult = dbe_->book1D("MuonMult", "Muon Multiplicity", 10, 0, 10);
  hMuon1Pt  = dbe_->book1D("Muon1Pt",  "Muon 1 Pt ",        100, 0, 100);
  hMuon2Pt  = dbe_->book1D("Muon2Pt",  "Muon 2 Pt ",        100, 0, 100);
  hMuon1Eta  = dbe_->book1D("Muon1Eta",  "Muon 1 Eta ",        10*binFactor, -3, 3);
  hMuon2Eta  = dbe_->book1D("Muon2Eta",  "Muon 2 Eta ",        10*binFactor, -3, 3);
  hMuon1Phi  = dbe_->book1D("Muon1Phi",  "Muon 1 Phi ",        10*binFactor, -3.2, 3.2);
  hMuon2Phi  = dbe_->book1D("Muon2Phi",  "Muon 2 Phi ",        10*binFactor, -3.2, 3.2);
  
  hDiMuonInvMass = dbe_->book1D("DiMuonInvMass", "DiMuon Invariant Mass", 100*binFactor, 0, 1000);

  dbe_->setCurrentFolder(dirname_+"/RecoMuons/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "MuonMult_" + (*l1Names_)[i];
    myHistoTitle = "Muon Multiplicity for L1 path " + (*l1Names_)[i];
    hMuonMultAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Muon1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Muon 1 Pt for L1 path " + (*l1Names_)[i];
    hMuon1PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Muon 2 Pt for L1 path " + (*l1Names_)[i];
    hMuon2PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Muon 1 Eta for L1 path " + (*l1Names_)[i];
    hMuon1EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Muon2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Muon 2 Eta for L1 path " + (*l1Names_)[i];
    hMuon2EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Muon1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Muon 1 Phi for L1 path " + (*l1Names_)[i];
    hMuon1PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));
    myHistoName = "Muon2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Muon 2 Phi for L1 path " + (*l1Names_)[i];
    hMuon2PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));

    myHistoName = "DiMuonInvMass_" + (*l1Names_)[i];
    myHistoTitle = "DiMuon Invariant Mass for L1 path " + (*l1Names_)[i];
    hDiMuonInvMassAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100*binFactor, 0, 1000));

  }

  dbe_->setCurrentFolder(dirname_+"/RecoMuons/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "MuonMult_" + (*hlNames_)[i];
    myHistoTitle = "Muon Multiplicity for HLT path " + (*hlNames_)[i];    
    hMuonMultAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Muon1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Muon 1 Pt for HLT path " + (*hlNames_)[i];
    hMuon1PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Muon 2 Pt for HLT path " + (*hlNames_)[i];
    hMuon2PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Muon 1 Eta for HLT path " + (*hlNames_)[i];
    hMuon1EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Muon2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Muon 2 Eta for HLT path " + (*hlNames_)[i];
    hMuon2EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Muon1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Muon 1 Phi for HLT path " + (*hlNames_)[i];
    hMuon1PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));
    myHistoName = "Muon2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Muon 2 Phi for HLT path " + (*hlNames_)[i];
    hMuon2PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));

    myHistoName = "DiMuonInvMass_" + (*hlNames_)[i];
    myHistoTitle = "DiMuon Invariant Mass for HLT path " + (*hlNames_)[i];
    hDiMuonInvMassAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100*binFactor, 0, 1000));

  }
  dbe_->setCurrentFolder(dirname_);



  //******************
  //Book Photons
  //******************
  
  dbe_->setCurrentFolder(dirname_+"/RecoPhotons/General");
  hPhotonMult = dbe_->book1D("PhotonMult", "Photon Multiplicity", 10, 0, 10);
  hPhoton1Pt  = dbe_->book1D("Photon1Pt",  "Photon 1 Pt ",        100, 0, 100);
  hPhoton2Pt  = dbe_->book1D("Photon2Pt",  "Photon 2 Pt ",        100, 0, 100);
  hPhoton1Eta  = dbe_->book1D("Photon1Eta",  "Photon 1 Eta ",        10*binFactor, -3, 3);
  hPhoton2Eta  = dbe_->book1D("Photon2Eta",  "Photon 2 Eta ",        10*binFactor, -3, 3);
  hPhoton1Phi  = dbe_->book1D("Photon1Phi",  "Photon 1 Phi ",        10*binFactor, -3.2, 3.2);
  hPhoton2Phi  = dbe_->book1D("Photon2Phi",  "Photon 2 Phi ",        10*binFactor, -3.2, 3.2);
  
  hDiPhotonInvMass = dbe_->book1D("DiPhotonInvMass", "DiPhoton Invariant Mass", 100*binFactor, 0, 1000);

  dbe_->setCurrentFolder(dirname_+"/RecoPhotons/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "PhotonMult_" + (*l1Names_)[i];
    myHistoTitle = "Photon Multiplicity for L1 path " + (*l1Names_)[i];
    hPhotonMultAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Photon1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Photon 1 Pt for L1 path " + (*l1Names_)[i];
    hPhoton1PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Photon2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Photon 2 Pt for L1 path " + (*l1Names_)[i];
    hPhoton2PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Photon1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Photon 1 Eta for L1 path " + (*l1Names_)[i];
    hPhoton1EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Photon2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Photon 2 Eta for L1 path " + (*l1Names_)[i];
    hPhoton2EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Photon1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Photon 1 Phi for L1 path " + (*l1Names_)[i];
    hPhoton1PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));
    myHistoName = "Photon2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Photon 2 Phi for L1 path " + (*l1Names_)[i];
    hPhoton2PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));

    myHistoName = "DiPhotonInvMass_" + (*l1Names_)[i];
    myHistoTitle = "DiPhoton Invariant Mass for L1 path " + (*l1Names_)[i];
    hDiPhotonInvMassAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100*binFactor, 0, 1000));

  }

  dbe_->setCurrentFolder(dirname_+"/RecoPhotons/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "PhotonMult_" + (*hlNames_)[i];
    myHistoTitle = "Photon Multiplicity for HLT path " + (*hlNames_)[i];    
    hPhotonMultAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Photon1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Photon 1 Pt for HLT path " + (*hlNames_)[i];
    hPhoton1PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Photon2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Photon 2 Pt for HLT path " + (*hlNames_)[i];
    hPhoton2PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Photon1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Photon 1 Eta for HLT path " + (*hlNames_)[i];
    hPhoton1EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Photon2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Photon 2 Eta for HLT path " + (*hlNames_)[i];
    hPhoton2EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3, 3));
    myHistoName = "Photon1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Photon 1 Phi for HLT path " + (*hlNames_)[i];
    hPhoton1PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));
    myHistoName = "Photon2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Photon 2 Phi for HLT path " + (*hlNames_)[i];
    hPhoton2PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));

    myHistoName = "DiPhotonInvMass_" + (*hlNames_)[i];
    myHistoTitle = "DiPhoton Invariant Mass for HLT path " + (*hlNames_)[i];
    hDiPhotonInvMassAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100*binFactor, 0, 1000));

  }
  dbe_->setCurrentFolder(dirname_);



  //******************
  //Book MET
  //******************
  
  dbe_->setCurrentFolder(dirname_+"/RecoMET/General");
  hMET = dbe_->book1D("MET", "MET", 35, 0, 1050);
  hMETx   = dbe_->book1D("METx", "METx", 35, 0, 1050);
  hMETy   = dbe_->book1D("METy", "METy", 35, 0, 1050);
  hMETphi = dbe_->book1D("METphi", "METphi", 10*binFactor, -3.2, 3.2);
  hSumEt  = dbe_->book1D("SumEt" , "SumEt",  35, 0, 1050);
  hMETSignificance = dbe_->book1D("METSignificance", "METSignificance", 100, 0, 100);
  dbe_->setCurrentFolder(dirname_+"/RecoMET/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "MET_" + (*l1Names_)[i];
    myHistoTitle = "MET for L1 path " + (*l1Names_)[i];
    hMETAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METx_" + (*l1Names_)[i];
    myHistoTitle = "METx for L1 path " + (*l1Names_)[i];
    hMETxAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METy_" + (*l1Names_)[i];
    myHistoTitle = "METy for L1 path " + (*l1Names_)[i];
    hMETyAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METPhi_" + (*l1Names_)[i];
    myHistoTitle = "METPhi for L1 path " + (*l1Names_)[i];
    hMETphiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2));
    myHistoName = "SumEt_" + (*l1Names_)[i];
    myHistoTitle = "SumEt for L1 path " + (*l1Names_)[i];
    hSumEtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METSignificance_" + (*l1Names_)[i];
    myHistoTitle = "METSignificance for L1 path " + (*l1Names_)[i];
    hMETSignificanceAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, 0, 100));
  }

  dbe_->setCurrentFolder(dirname_+"/RecoMET/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "MET_" + (*hlNames_)[i];
    myHistoTitle = "MET for HLT path " + (*hlNames_)[i];    
    hMETAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METx_" + (*hlNames_)[i];
    myHistoTitle = "METx for HLT path " + (*hlNames_)[i];    
    hMETxAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METy_" + (*hlNames_)[i];
    myHistoTitle = "METy for HLT path " + (*hlNames_)[i];    
    hMETyAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METPhi_" + (*hlNames_)[i];
    myHistoTitle = "METPhi for HLT path " + (*hlNames_)[i];    
    hMETphiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, -3.2, 3.2 ));
    myHistoName = "SumEt_" + (*hlNames_)[i];
    myHistoTitle = "SumEt for HLT path " + (*hlNames_)[i];    
    hSumEtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METSignificance_" + (*hlNames_)[i];
    myHistoTitle = "METSignificance for HLT path " + (*hlNames_)[i];    
    hMETSignificanceAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10*binFactor, 0, 100));
  }
  dbe_->setCurrentFolder(dirname_);




}






void PlotMakerReco::handleObjects(const edm::Event& iEvent)
{


  //***********************************************
  // Get the RECO Objects
  //***********************************************


  //Get the electrons
  Handle<GsfElectronCollection> theElectronCollectionHandle; 
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

double PlotMakerReco::invariantMass(reco::Candidate* p1, reco::Candidate* p2) {
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

