/*  \class PlotMakerL1
*
*  Class to produce some plots of Off-line variables in the TriggerValidation Code
*
*  Author: Massimiliano Chiorboli      Date: September 2007
//         Maurizio Pierini
//         Maria Spiropulu
*
*/
#include "HLTriggerOffline/SUSYBSM/interface/PlotMakerL1.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TDirectory.h"

#include "HLTriggerOffline/SUSYBSM/interface/PtSorter.h"


using namespace edm;
using namespace reco;
using namespace std;
using namespace l1extra;

PlotMakerL1::PlotMakerL1(edm::ParameterSet PlotMakerL1Input)
{
  m_l1extra      	 = PlotMakerL1Input.getParameter<string>("l1extramc");

  dirname_          = PlotMakerL1Input.getParameter<std::string>("dirname");

}

void PlotMakerL1::fillPlots(const edm::Event& iEvent)
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

}




void PlotMakerL1::bookHistos(DQMStore * dbe_, std::vector<int>* l1bits, std::vector<int>* hltbits, 
			   std::vector<std::string>* l1Names_, std::vector<std::string>* hlNames_)
{

  this->setBits(l1bits, hltbits);

  //******************
  //Book histos for L1 Objects
  //******************
  
  
  //******************
  //Book Jets
  //******************
  //  std::string dirname_="HLTOffline/TriggerValidator/"; 

  dbe_->setCurrentFolder(dirname_+"/L1Jets/Central/General");
  hL1CentralJetMult = dbe_->book1D("JetMult", "Jet Multiplicity", 10, 0, 10);
  hL1CentralJet1Pt  = dbe_->book1D("Jet1Pt",  "Jet 1 Pt ",        100, 0, 1000);
  hL1CentralJet2Pt  = dbe_->book1D("Jet2Pt",  "Jet 2 Pt ",        100, 0, 1000);
  hL1CentralJet1Eta  = dbe_->book1D("Jet1Eta",  "Jet 1 Eta ",     100, -3, 3);
  hL1CentralJet2Eta  = dbe_->book1D("Jet2Eta",  "Jet 2 Eta ",     100, -3, 3);
  hL1CentralJet1Phi  = dbe_->book1D("Jet1Phi",  "Jet 1 Phi ",     100, -3.2, 3.2);
  hL1CentralJet2Phi  = dbe_->book1D("Jet2Phi",  "Jet 2 Phi ",     100, -3.2, 3.2);

  dbe_->setCurrentFolder(dirname_+"/L1Jets/Central/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "JetMult_" + (*l1Names_)[i];
    myHistoTitle = "Jet Multiplicity for L1 path " + (*l1Names_)[i];
    hL1CentralJetMultAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Pt for L1 path " + (*l1Names_)[i];
    hL1CentralJet1PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Pt for L1 path " + (*l1Names_)[i];
    hL1CentralJet2PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Eta for L1 path " + (*l1Names_)[i];
    hL1CentralJet1EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Eta for L1 path " + (*l1Names_)[i];
    hL1CentralJet2EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Phi for L1 path " + (*l1Names_)[i];
   hL1CentralJet1PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Phi for L1 path " + (*l1Names_)[i];
    hL1CentralJet2PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  dbe_->setCurrentFolder(dirname_+"/L1Jets/Central/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "JetMult_" + (*hlNames_)[i];
    myHistoTitle = "Jet Multiplicity for HLT path " + (*hlNames_)[i];
    hL1CentralJetMultAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Pt for HLT path " + (*hlNames_)[i];
    hL1CentralJet1PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Pt for HLT path " + (*hlNames_)[i];
    hL1CentralJet2PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Eta for HLT path " + (*hlNames_)[i];
    hL1CentralJet1EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Eta for HLT path " + (*hlNames_)[i];
    hL1CentralJet2EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Phi for HLT path " + (*hlNames_)[i];
    hL1CentralJet1PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Phi for HLT path " + (*hlNames_)[i];
    hL1CentralJet2PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  dbe_->setCurrentFolder(dirname_+"/L1Jets/Forward/General");
  hL1ForwardJetMult = dbe_->book1D("JetMult", "Jet Multiplicity", 10, 0, 10);
  hL1ForwardJet1Pt  = dbe_->book1D("Jet1Pt",  "Jet 1 Pt ",        100, 0, 1000);
  hL1ForwardJet2Pt  = dbe_->book1D("Jet2Pt",  "Jet 2 Pt ",        100, 0, 1000);
  hL1ForwardJet1Eta  = dbe_->book1D("Jet1Eta",  "Jet 1 Eta ",        100, -3, 3);
  hL1ForwardJet2Eta  = dbe_->book1D("Jet2Eta",  "Jet 2 Eta ",        100, -3, 3);
  hL1ForwardJet1Phi  = dbe_->book1D("Jet1Phi",  "Jet 1 Phi ",        100, -3.2, 3.2);
  hL1ForwardJet2Phi  = dbe_->book1D("Jet2Phi",  "Jet 2 Phi ",        100, -3.2, 3.2);


  dbe_->setCurrentFolder(dirname_+"/L1Jets/Forward/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "JetMult_" + (*l1Names_)[i];
    myHistoTitle = "Jet Multiplicity for L1 path " + (*l1Names_)[i];
    hL1ForwardJetMultAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Pt for L1 path " + (*l1Names_)[i];
    hL1ForwardJet1PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Pt for L1 path " + (*l1Names_)[i];
    hL1ForwardJet2PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Eta for L1 path " + (*l1Names_)[i];
    hL1ForwardJet1EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Eta for L1 path " + (*l1Names_)[i];
    hL1ForwardJet2EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Phi for L1 path " + (*l1Names_)[i];
    hL1ForwardJet1PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Phi for L1 path " + (*l1Names_)[i];
    hL1ForwardJet2PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  dbe_->setCurrentFolder(dirname_+"/L1Jets/Forward/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "JetMult_" + (*hlNames_)[i];
    myHistoTitle = "Jet Multiplicity for HLT path " + (*hlNames_)[i];
    hL1ForwardJetMultAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Pt for HLT path " + (*hlNames_)[i];
    hL1ForwardJet1PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Pt for HLT path " + (*hlNames_)[i];
    hL1ForwardJet2PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Eta for HLT path " + (*hlNames_)[i];
    hL1ForwardJet1EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Eta for HLT path " + (*hlNames_)[i];
    hL1ForwardJet2EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Phi for HLT path " + (*hlNames_)[i];
    hL1ForwardJet1PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Phi for HLT path " + (*hlNames_)[i];
    hL1ForwardJet2PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }


  dbe_->setCurrentFolder(dirname_+"/L1Jets/Tau/General");
  hL1TauJetMult = dbe_->book1D("JetMult", "Jet Multiplicity", 10, 0, 10);
  hL1TauJet1Pt  = dbe_->book1D("Jet1Pt",  "Jet 1 Pt ",        100, 0, 1000);
  hL1TauJet2Pt  = dbe_->book1D("Jet2Pt",  "Jet 2 Pt ",        100, 0, 1000);
  hL1TauJet1Eta  = dbe_->book1D("Jet1Eta",  "Jet 1 Eta ",        100, -3, 3);
  hL1TauJet2Eta  = dbe_->book1D("Jet2Eta",  "Jet 2 Eta ",        100, -3, 3);
  hL1TauJet1Phi  = dbe_->book1D("Jet1Phi",  "Jet 1 Phi ",        100, -3.2, 3.2);
  hL1TauJet2Phi  = dbe_->book1D("Jet2Phi",  "Jet 2 Phi ",        100, -3.2, 3.2);

  dbe_->setCurrentFolder(dirname_+"/L1Jets/Tau/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "JetMult_" + (*l1Names_)[i];
    myHistoTitle = "Jet Multiplicity for L1 path " + (*l1Names_)[i];
    hL1TauJetMultAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Pt for L1 path " + (*l1Names_)[i];
    hL1TauJet1PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Pt for L1 path " + (*l1Names_)[i];
    hL1TauJet2PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Eta for L1 path " + (*l1Names_)[i];
    hL1TauJet1EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Eta for L1 path " + (*l1Names_)[i];
    hL1TauJet2EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 1 Phi for L1 path " + (*l1Names_)[i];
    hL1TauJet1PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Jet 2 Phi for L1 path " + (*l1Names_)[i];
    hL1TauJet2PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  dbe_->setCurrentFolder(dirname_+"/L1Jets/Tau/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "JetMult_" + (*hlNames_)[i];
    myHistoTitle = "Jet Multiplicity for HLT path " + (*hlNames_)[i];
    hL1TauJetMultAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Jet1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Pt for HLT path " + (*hlNames_)[i];
    hL1TauJet1PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Pt for HLT path " + (*hlNames_)[i];
    hL1TauJet2PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
    myHistoName = "Jet1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Eta for HLT path " + (*hlNames_)[i];
    hL1TauJet1EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Eta for HLT path " + (*hlNames_)[i];
    hL1TauJet2EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Jet1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 1 Phi for HLT path " + (*hlNames_)[i];
    hL1TauJet1PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Jet2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Jet 2 Phi for HLT path " + (*hlNames_)[i];
    hL1TauJet2PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }



  dbe_->setCurrentFolder(dirname_+"/L1Em/Isolated/General");
  hL1EmIsoMult = dbe_->book1D("ElecMult", "Elec Multiplicity", 10, 0, 10);
  hL1EmIso1Pt  = dbe_->book1D("Elec1Pt",  "Elec 1 Pt ",        100, 0, 100);
  hL1EmIso2Pt  = dbe_->book1D("Elec2Pt",  "Elec 2 Pt ",        100, 0, 100);
  hL1EmIso1Eta  = dbe_->book1D("Elec1Eta",  "Elec 1 Eta ",        100, -3, 3);
  hL1EmIso2Eta  = dbe_->book1D("Elec2Eta",  "Elec 2 Eta ",        100, -3, 3);
  hL1EmIso1Phi  = dbe_->book1D("Elec1Phi",  "Elec 1 Phi ",        100, -3.2, 3.2);
  hL1EmIso2Phi  = dbe_->book1D("Elec2Phi",  "Elec 2 Phi ",        100, -3.2, 3.2);
  
  dbe_->setCurrentFolder(dirname_+"/L1Em/Isolated/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "ElecMult_" + (*l1Names_)[i];
    myHistoTitle = "Elec Multiplicity for L1 path " + (*l1Names_)[i];
    hL1EmIsoMultAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Elec1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Pt for L1 path " + (*l1Names_)[i];
    hL1EmIso1PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Pt for L1 path " + (*l1Names_)[i];
    hL1EmIso2PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Eta for L1 path " + (*l1Names_)[i];
    hL1EmIso1EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Eta for L1 path " + (*l1Names_)[i];
    hL1EmIso2EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Phi for L1 path " + (*l1Names_)[i];
    hL1EmIso1PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Elec2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Phi for L1 path " + (*l1Names_)[i];
    hL1EmIso2PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  dbe_->setCurrentFolder(dirname_+"/L1Em/Isolated/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "ElecMult_" + (*hlNames_)[i];
    myHistoTitle = "Elec Multiplicity for HLT path " + (*hlNames_)[i];    
    hL1EmIsoMultAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Elec1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Pt for HLT path " + (*hlNames_)[i];
    hL1EmIso1PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Pt for HLT path " + (*hlNames_)[i];
    hL1EmIso2PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Eta for HLT path " + (*hlNames_)[i];
    hL1EmIso1EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Eta for HLT path " + (*hlNames_)[i];
    hL1EmIso2EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Phi for HLT path " + (*hlNames_)[i];
    hL1EmIso1PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Elec2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Phi for HLT path " + (*hlNames_)[i];
    hL1EmIso2PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }
  dbe_->setCurrentFolder(dirname_);



  dbe_->setCurrentFolder(dirname_+"/L1Em/NotIsolated/General");
  hL1EmNotIsoMult = dbe_->book1D("ElecMult", "Elec Multiplicity", 10, 0, 10);
  hL1EmNotIso1Pt  = dbe_->book1D("Elec1Pt",  "Elec 1 Pt ",        100, 0, 100);
  hL1EmNotIso2Pt  = dbe_->book1D("Elec2Pt",  "Elec 2 Pt ",        100, 0, 100);
  hL1EmNotIso1Eta  = dbe_->book1D("Elec1Eta",  "Elec 1 Eta ",        100, -3, 3);
  hL1EmNotIso2Eta  = dbe_->book1D("Elec2Eta",  "Elec 2 Eta ",        100, -3, 3);
  hL1EmNotIso1Phi  = dbe_->book1D("Elec1Phi",  "Elec 1 Phi ",        100, -3.2, 3.2);
  hL1EmNotIso2Phi  = dbe_->book1D("Elec2Phi",  "Elec 2 Phi ",        100, -3.2, 3.2);
  
  dbe_->setCurrentFolder(dirname_+"/L1Em/NotIsolated/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "ElecMult_" + (*l1Names_)[i];
    myHistoTitle = "Elec Multiplicity for L1 path " + (*l1Names_)[i];
    hL1EmNotIsoMultAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Elec1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Pt for L1 path " + (*l1Names_)[i];
    hL1EmNotIso1PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Pt for L1 path " + (*l1Names_)[i];
    hL1EmNotIso2PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Eta for L1 path " + (*l1Names_)[i];
    hL1EmNotIso1EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Eta for L1 path " + (*l1Names_)[i];
    hL1EmNotIso2EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Elec 1 Phi for L1 path " + (*l1Names_)[i];
    hL1EmNotIso1PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Elec2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Elec 2 Phi for L1 path " + (*l1Names_)[i];
    hL1EmNotIso2PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
   }

  dbe_->setCurrentFolder(dirname_+"/L1Em/NotIsolated/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "ElecMult_" + (*hlNames_)[i];
    myHistoTitle = "Elec Multiplicity for HLT path " + (*hlNames_)[i];    
    hL1EmNotIsoMultAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Elec1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Pt for HLT path " + (*hlNames_)[i];
    hL1EmNotIso1PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Pt for HLT path " + (*hlNames_)[i];
    hL1EmNotIso2PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Elec1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Eta for HLT path " + (*hlNames_)[i];
    hL1EmNotIso1EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Eta for HLT path " + (*hlNames_)[i];
    hL1EmNotIso2EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Elec1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Elec 1 Phi for HLT path " + (*hlNames_)[i];
    hL1EmNotIso1PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Elec2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Elec 2 Phi for HLT path " + (*hlNames_)[i];
    hL1EmNotIso2PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }
  dbe_->setCurrentFolder(dirname_);

  //******************
  //Book Muons
  //******************
  
  dbe_->setCurrentFolder(dirname_+"/L1Muons/General");
  hL1MuonMult = dbe_->book1D("MuonMult", "Muon Multiplicity", 10, 0, 10);
  hL1Muon1Pt  = dbe_->book1D("Muon1Pt",  "Muon 1 Pt ",        100, 0, 100);
  hL1Muon2Pt  = dbe_->book1D("Muon2Pt",  "Muon 2 Pt ",        100, 0, 100);
  hL1Muon1Eta  = dbe_->book1D("Muon1Eta",  "Muon 1 Eta ",        100, -3, 3);
  hL1Muon2Eta  = dbe_->book1D("Muon2Eta",  "Muon 2 Eta ",        100, -3, 3);
  hL1Muon1Phi  = dbe_->book1D("Muon1Phi",  "Muon 1 Phi ",        100, -3.2, 3.2);
  hL1Muon2Phi  = dbe_->book1D("Muon2Phi",  "Muon 2 Phi ",        100, -3.2, 3.2);
  
  dbe_->setCurrentFolder(dirname_+"/L1Muons/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "MuonMult_" + (*l1Names_)[i];
    myHistoTitle = "Muon Multiplicity for L1 path " + (*l1Names_)[i];
    hL1MuonMultAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Muon1Pt_" + (*l1Names_)[i];
    myHistoTitle = "Muon 1 Pt for L1 path " + (*l1Names_)[i];
    hL1Muon1PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon2Pt_" + (*l1Names_)[i];
    myHistoTitle = "Muon 2 Pt for L1 path " + (*l1Names_)[i];
    hL1Muon2PtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon1Eta_" + (*l1Names_)[i];
    myHistoTitle = "Muon 1 Eta for L1 path " + (*l1Names_)[i];
    hL1Muon1EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Muon2Eta_" + (*l1Names_)[i];
    myHistoTitle = "Muon 2 Eta for L1 path " + (*l1Names_)[i];
    hL1Muon2EtaAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Muon1Phi_" + (*l1Names_)[i];
    myHistoTitle = "Muon 1 Phi for L1 path " + (*l1Names_)[i];
    hL1Muon1PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Muon2Phi_" + (*l1Names_)[i];
    myHistoTitle = "Muon 2 Phi for L1 path " + (*l1Names_)[i];
    hL1Muon2PhiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }

  dbe_->setCurrentFolder(dirname_+"/L1Muons/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "MuonMult_" + (*hlNames_)[i];
    myHistoTitle = "Muon Multiplicity for HLT path " + (*hlNames_)[i];    
    hL1MuonMultAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 10, 0, 10));
    myHistoName = "Muon1Pt_" + (*hlNames_)[i];
    myHistoTitle = "Muon 1 Pt for HLT path " + (*hlNames_)[i];
    hL1Muon1PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon2Pt_" + (*hlNames_)[i];
    myHistoTitle = "Muon 2 Pt for HLT path " + (*hlNames_)[i];
    hL1Muon2PtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
    myHistoName = "Muon1Eta_" + (*hlNames_)[i];
    myHistoTitle = "Muon 1 Eta for HLT path " + (*hlNames_)[i];
    hL1Muon1EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Muon2Eta_" + (*hlNames_)[i];
    myHistoTitle = "Muon 2 Eta for HLT path " + (*hlNames_)[i];
    hL1Muon2EtaAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3, 3));
    myHistoName = "Muon1Phi_" + (*hlNames_)[i];
    myHistoTitle = "Muon 1 Phi for HLT path " + (*hlNames_)[i];
    hL1Muon1PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "Muon2Phi_" + (*hlNames_)[i];
    myHistoTitle = "Muon 2 Phi for HLT path " + (*hlNames_)[i];
    hL1Muon2PhiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
  }
  dbe_->setCurrentFolder(dirname_);



  //******************
  //Book MET
  //******************
  
  dbe_->setCurrentFolder(dirname_+"/L1MET/General");
  hL1MET = dbe_->book1D("MET", "MET", 35, 0, 1050);
  hL1METx   = dbe_->book1D("METx", "METx", 35, 0, 1050);
  hL1METy   = dbe_->book1D("METy", "METy", 35, 0, 1050);
  hL1METphi = dbe_->book1D("METphi", "METphi", 100, -3.2, 3.2);
  hL1SumEt  = dbe_->book1D("SumEt", "SumEt", 35, 0, 1050);
  hL1METSignificance = dbe_->book1D("METSignificance", "METSignificance", 100, 0, 100);


  dbe_->setCurrentFolder(dirname_+"/L1MET/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "MET_" + (*l1Names_)[i];
    myHistoTitle = "MET for L1 path " + (*l1Names_)[i];
    hL1METAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METx_" + (*l1Names_)[i];
    myHistoTitle = "METx for L1 path " + (*l1Names_)[i];
    hL1METxAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METy_" + (*l1Names_)[i];
    myHistoTitle = "METy for L1 path " + (*l1Names_)[i];
    hL1METyAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METphi_" + (*l1Names_)[i];
    myHistoTitle = "METphi for L1 path " + (*l1Names_)[i];
    hL1METphiAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "SumEt_" + (*l1Names_)[i];
    myHistoTitle = "SumEt for L1 path " + (*l1Names_)[i];
    hL1SumEtAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METSignificance_" + (*l1Names_)[i];
    myHistoTitle = "METSignificance for L1 path " + (*l1Names_)[i];
    hL1METSignificanceAfterL1.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
  }

  dbe_->setCurrentFolder(dirname_+"/L1MET/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "MET_" + (*hlNames_)[i];
    myHistoTitle = "MET for HLT path " + (*hlNames_)[i];    
    hL1METAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METx_" + (*hlNames_)[i];
    myHistoTitle = "METx for HLT path " + (*hlNames_)[i];    
    hL1METxAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METy_" + (*hlNames_)[i];
    myHistoTitle = "METy for HLT path " + (*hlNames_)[i];    
    hL1METyAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METphi_" + (*hlNames_)[i];
    myHistoTitle = "METphi for HLT path " + (*hlNames_)[i];    
    hL1METphiAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, -3.2, 3.2));
    myHistoName = "SumEt_" + (*hlNames_)[i];
    myHistoTitle = "SumEt for HLT path " + (*hlNames_)[i];    
    hL1SumEtAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 35, 0, 1050));
    myHistoName = "METSignificance_" + (*hlNames_)[i];
    myHistoTitle = "METSignificance for HLT path " + (*hlNames_)[i];    
    hL1METSignificanceAfterHLT.push_back(dbe_->book1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 100));
  }
  dbe_->setCurrentFolder(dirname_);



}





void PlotMakerL1::handleObjects(const edm::Event& iEvent)
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
  iEvent.getByLabel(m_l1extra,"MET",theL1METHandle);
  //iEvent.getByLabel(m_l1extra,theL1METHandle);
  theL1METCollection = *theL1METHandle;
  std::sort(theL1METCollection.begin(), theL1METCollection.end(),PtSorter());

}

double PlotMakerL1::invariantMass(reco::Candidate* p1, reco::Candidate* p2) {
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

