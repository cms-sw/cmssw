/*  \class PlotMaker
*
*  Class to produce some plots of Off-line variables in the TriggerValidation Code
*
*  Author: Massimiliano Chiorboli      Date: September 2007
*
*/
#include "HLTriggerOffline/SUSYBSM/interface/PlotMaker.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TDirectory.h"

using namespace edm;
using namespace reco;
using namespace std;

PlotMaker::PlotMaker(edm::ParameterSet objectList)
{
  m_l1extra      = objectList.getParameter<string>("l1objects");
  m_electronSrc  = objectList.getParameter<string>("electrons");
  m_muonSrc    	 = objectList.getParameter<string>("muons");
  m_jetsSrc    	 = objectList.getParameter<string>("jets");
  m_photonSrc  	 = objectList.getParameter<string>("photons");
  m_calometSrc 	 = objectList.getParameter<string>("calomet");

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

  hL1CentralJetMult->Fill(theL1CentralJetCollection->size());
  if(theL1CentralJetCollection->size()>0) hL1CentralJet1Pt->Fill((*theL1CentralJetCollection)[0].pt());
  if(theL1CentralJetCollection->size()>1) hL1CentralJet1Pt->Fill((*theL1CentralJetCollection)[1].pt());
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1CentralJetMultAfterL1[i]->Fill(theL1CentralJetCollection->size());
      if(theL1CentralJetCollection->size()>0) hL1CentralJet1PtAfterL1[i]->Fill((*theL1CentralJetCollection)[0].pt());
      if(theL1CentralJetCollection->size()>1) hL1CentralJet1PtAfterL1[i]->Fill((*theL1CentralJetCollection)[1].pt());
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1CentralJetMultAfterHLT[i]->Fill(theL1CentralJetCollection->size());
      if(theL1CentralJetCollection->size()>0) hL1CentralJet1PtAfterHLT[i]->Fill((*theL1CentralJetCollection)[0].pt());
      if(theL1CentralJetCollection->size()>1) hL1CentralJet1PtAfterHLT[i]->Fill((*theL1CentralJetCollection)[1].pt());
    }
  }



  hL1ForwardJetMult->Fill(theL1ForwardJetCollection->size());
  if(theL1ForwardJetCollection->size()>0) hL1ForwardJet1Pt->Fill((*theL1ForwardJetCollection)[0].pt());
  if(theL1ForwardJetCollection->size()>1) hL1ForwardJet1Pt->Fill((*theL1ForwardJetCollection)[1].pt());
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1ForwardJetMultAfterL1[i]->Fill(theL1ForwardJetCollection->size());
      if(theL1ForwardJetCollection->size()>0) hL1ForwardJet1PtAfterL1[i]->Fill((*theL1ForwardJetCollection)[0].pt());
      if(theL1ForwardJetCollection->size()>1) hL1ForwardJet1PtAfterL1[i]->Fill((*theL1ForwardJetCollection)[1].pt());
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1ForwardJetMultAfterHLT[i]->Fill(theL1ForwardJetCollection->size());
      if(theL1ForwardJetCollection->size()>0) hL1ForwardJet1PtAfterHLT[i]->Fill((*theL1ForwardJetCollection)[0].pt());
      if(theL1ForwardJetCollection->size()>1) hL1ForwardJet1PtAfterHLT[i]->Fill((*theL1ForwardJetCollection)[1].pt());
    }
  }




  hL1TauJetMult->Fill(theL1TauJetCollection->size());
  if(theL1TauJetCollection->size()>0) hL1TauJet1Pt->Fill((*theL1TauJetCollection)[0].pt());
  if(theL1TauJetCollection->size()>1) hL1TauJet1Pt->Fill((*theL1TauJetCollection)[1].pt());
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1TauJetMultAfterL1[i]->Fill(theL1TauJetCollection->size());
      if(theL1TauJetCollection->size()>0) hL1TauJet1PtAfterL1[i]->Fill((*theL1TauJetCollection)[0].pt());
      if(theL1TauJetCollection->size()>1) hL1TauJet1PtAfterL1[i]->Fill((*theL1TauJetCollection)[1].pt());
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1TauJetMultAfterHLT[i]->Fill(theL1TauJetCollection->size());
      if(theL1TauJetCollection->size()>0) hL1TauJet1PtAfterHLT[i]->Fill((*theL1TauJetCollection)[0].pt());
      if(theL1TauJetCollection->size()>1) hL1TauJet1PtAfterHLT[i]->Fill((*theL1TauJetCollection)[1].pt());
    }
  }



  hL1EmIsoMult->Fill(theL1EmIsoCollection->size());
  if(theL1EmIsoCollection->size()>0) hL1EmIso1Pt->Fill((*theL1EmIsoCollection)[0].pt());
  if(theL1EmIsoCollection->size()>1) hL1EmIso1Pt->Fill((*theL1EmIsoCollection)[1].pt());
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1EmIsoMultAfterL1[i]->Fill(theL1EmIsoCollection->size());
      if(theL1EmIsoCollection->size()>0) hL1EmIso1PtAfterL1[i]->Fill((*theL1EmIsoCollection)[0].pt());
      if(theL1EmIsoCollection->size()>1) hL1EmIso1PtAfterL1[i]->Fill((*theL1EmIsoCollection)[1].pt());
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1EmIsoMultAfterHLT[i]->Fill(theL1EmIsoCollection->size());
      if(theL1EmIsoCollection->size()>0) hL1EmIso1PtAfterHLT[i]->Fill((*theL1EmIsoCollection)[0].pt());
      if(theL1EmIsoCollection->size()>1) hL1EmIso1PtAfterHLT[i]->Fill((*theL1EmIsoCollection)[1].pt());
    }
  }

  hL1EmNotIsoMult->Fill(theL1EmNotIsoCollection->size());
  if(theL1EmNotIsoCollection->size()>0) hL1EmNotIso1Pt->Fill((*theL1EmNotIsoCollection)[0].pt());
  if(theL1EmNotIsoCollection->size()>1) hL1EmNotIso1Pt->Fill((*theL1EmNotIsoCollection)[1].pt());
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1EmNotIsoMultAfterL1[i]->Fill(theL1EmNotIsoCollection->size());
      if(theL1EmNotIsoCollection->size()>0) hL1EmNotIso1PtAfterL1[i]->Fill((*theL1EmNotIsoCollection)[0].pt());
      if(theL1EmNotIsoCollection->size()>1) hL1EmNotIso1PtAfterL1[i]->Fill((*theL1EmNotIsoCollection)[1].pt());
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1EmNotIsoMultAfterHLT[i]->Fill(theL1EmNotIsoCollection->size());
      if(theL1EmNotIsoCollection->size()>0) hL1EmNotIso1PtAfterHLT[i]->Fill((*theL1EmNotIsoCollection)[0].pt());
      if(theL1EmNotIsoCollection->size()>1) hL1EmNotIso1PtAfterHLT[i]->Fill((*theL1EmNotIsoCollection)[1].pt());
    }
  }





  //**********************
  // Fill the Muon Histos
  //**********************
  
  hL1MuonMult->Fill(theL1MuonCollection->size());
  if(theL1MuonCollection->size()>0) hL1Muon1Pt->Fill((*theL1MuonCollection)[0].pt());
  if(theL1MuonCollection->size()>1) hL1Muon1Pt->Fill((*theL1MuonCollection)[1].pt());
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hL1MuonMultAfterL1[i]->Fill(theL1MuonCollection->size());
      if(theL1MuonCollection->size()>0) hL1Muon1PtAfterL1[i]->Fill((*theL1MuonCollection)[0].pt());
      if(theL1MuonCollection->size()>1) hL1Muon1PtAfterL1[i]->Fill((*theL1MuonCollection)[1].pt());
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hL1MuonMultAfterHLT[i]->Fill(theL1MuonCollection->size());
      if(theL1MuonCollection->size()>0) hL1Muon1PtAfterHLT[i]->Fill((*theL1MuonCollection)[0].pt());
      if(theL1MuonCollection->size()>1) hL1Muon1PtAfterHLT[i]->Fill((*theL1MuonCollection)[1].pt());
    }
  }


  //**********************
  // Fill the MET Histos
  //**********************
  
  hL1MET->Fill(theL1METCollection->etMiss());
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) hL1METAfterL1[i]->Fill(theL1METCollection->etMiss());
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) hL1METAfterHLT[i]->Fill(theL1METCollection->etMiss());
  }

  //**********************
  // Fill the Reco Object Histos
  //**********************

  //**********************
  // Fill the Jet Histos
  //**********************
  
  hJetMult->Fill(theCaloJetCollection->size());
  if(theCaloJetCollection->size()>0) hJet1Pt->Fill((*theCaloJetCollection)[0].pt());
  if(theCaloJetCollection->size()>1) hJet1Pt->Fill((*theCaloJetCollection)[1].pt());
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hJetMultAfterL1[i]->Fill(theCaloJetCollection->size());
      if(theCaloJetCollection->size()>0) hJet1PtAfterL1[i]->Fill((*theCaloJetCollection)[0].pt());
      if(theCaloJetCollection->size()>1) hJet1PtAfterL1[i]->Fill((*theCaloJetCollection)[1].pt());
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hJetMultAfterHLT[i]->Fill(theCaloJetCollection->size());
      if(theCaloJetCollection->size()>0) hJet1PtAfterHLT[i]->Fill((*theCaloJetCollection)[0].pt());
      if(theCaloJetCollection->size()>1) hJet1PtAfterHLT[i]->Fill((*theCaloJetCollection)[1].pt());
    }
  }


  //**********************
  // Fill the Electron Histos
  //**********************
  
  hElecMult->Fill(theElectronCollection->size());
  if(theElectronCollection->size()>0) hElec1Pt->Fill((*theElectronCollection)[0].pt());
  if(theElectronCollection->size()>1) hElec1Pt->Fill((*theElectronCollection)[1].pt());
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hElecMultAfterL1[i]->Fill(theElectronCollection->size());
      if(theElectronCollection->size()>0) hElec1PtAfterL1[i]->Fill((*theElectronCollection)[0].pt());
      if(theElectronCollection->size()>1) hElec1PtAfterL1[i]->Fill((*theElectronCollection)[1].pt());
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hElecMultAfterHLT[i]->Fill(theElectronCollection->size());
      if(theElectronCollection->size()>0) hElec1PtAfterHLT[i]->Fill((*theElectronCollection)[0].pt());
      if(theElectronCollection->size()>1) hElec1PtAfterHLT[i]->Fill((*theElectronCollection)[1].pt());
    }
  }


  //**********************
  // Fill the Muon Histos
  //**********************
  
  hMuonMult->Fill(theMuonCollection->size());
  if(theMuonCollection->size()>0) hMuon1Pt->Fill((*theMuonCollection)[0].pt());
  if(theMuonCollection->size()>1) hMuon1Pt->Fill((*theMuonCollection)[1].pt());
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) {
      hMuonMultAfterL1[i]->Fill(theMuonCollection->size());
      if(theMuonCollection->size()>0) hMuon1PtAfterL1[i]->Fill((*theMuonCollection)[0].pt());
      if(theMuonCollection->size()>1) hMuon1PtAfterL1[i]->Fill((*theMuonCollection)[1].pt());
    }
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) {
      hMuonMultAfterHLT[i]->Fill(theMuonCollection->size());
      if(theMuonCollection->size()>0) hMuon1PtAfterHLT[i]->Fill((*theMuonCollection)[0].pt());
      if(theMuonCollection->size()>1) hMuon1PtAfterHLT[i]->Fill((*theMuonCollection)[1].pt());
    }
  }

  //**********************
  // Fill the MET Histos
  //**********************
  
  hMET->Fill((theCaloMETCollection->front()).pt());
  for(unsigned int i=0; i<l1bits_->size(); i++) {
    if(l1bits_->at(i)) hMETAfterL1[i]->Fill((theCaloMETCollection->front()).pt());
  }
  for(unsigned int i=0; i<hltbits_->size(); i++) {
    if(hltbits_->at(i)) hMETAfterHLT[i]->Fill((theCaloMETCollection->front()).pt());
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
  hL1CentralJetMult = new TH1D("JetMult", "Jet Multiplicity", 30, 0, 30);
  hL1CentralJet1Pt  = new TH1D("Jet1Pt",  "Jet 1 Pt ",        100, 0, 1000);
  hL1CentralJet2Pt  = new TH1D("Jet2Pt",  "Jet 2 Pt ",        100, 0, 1000);
  hL1CentralJet1Eta  = new TH1D("Jet1Eta",  "Jet 1 Eta ",     100, -3, 3);
  hL1CentralJet2Eta  = new TH1D("Jet2Eta",  "Jet 2 Eta ",     100, -3, 3);

  gDirectory->cd("/L1Jets/Central/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "JetMult_" + (*l1Names_)[i];
    myHistoTitle = "Jet Multiplicity for L1 path " + (*l1Names_)[i];
    hL1CentralJetMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 20, 0, 20));
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
  }

  gDirectory->cd("/L1Jets/Forward/General");
  hL1ForwardJetMult = new TH1D("JetMult", "Jet Multiplicity", 30, 0, 30);
  hL1ForwardJet1Pt  = new TH1D("Jet1Pt",  "Jet 1 Pt ",        100, 0, 1000);
  hL1ForwardJet2Pt  = new TH1D("Jet2Pt",  "Jet 2 Pt ",        100, 0, 1000);
  hL1ForwardJet1Eta  = new TH1D("Jet1Eta",  "Jet 1 Eta ",        100, -3, 3);
  hL1ForwardJet2Eta  = new TH1D("Jet2Eta",  "Jet 2 Eta ",        100, -3, 3);

  gDirectory->cd("/L1Jets/Forward/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "JetMult_" + (*l1Names_)[i];
    myHistoTitle = "Jet Multiplicity for L1 path " + (*l1Names_)[i];
    hL1ForwardJetMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 20, 0, 20));
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
  }

  gDirectory->cd("/L1Jets/Tau/General");
  hL1TauJetMult = new TH1D("JetMult", "Jet Multiplicity", 30, 0, 30);
  hL1TauJet1Pt  = new TH1D("Jet1Pt",  "Jet 1 Pt ",        100, 0, 1000);
  hL1TauJet2Pt  = new TH1D("Jet2Pt",  "Jet 2 Pt ",        100, 0, 1000);
  hL1TauJet1Eta  = new TH1D("Jet1Eta",  "Jet 1 Eta ",        100, -3, -3);
  hL1TauJet2Eta  = new TH1D("Jet2Eta",  "Jet 2 Eta ",        100, -3, -3);

  gDirectory->cd("/L1Jets/Tau/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "JetMult_" + (*l1Names_)[i];
    myHistoTitle = "Jet Multiplicity for L1 path " + (*l1Names_)[i];
    hL1TauJetMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 20, 0, 20));
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
  }




  gDirectory->cd("/L1Em/Isolated/General");
  hL1EmIsoMult = new TH1D("ElecMult", "Elec Multiplicity", 10, 0, 10);
  hL1EmIso1Pt  = new TH1D("Elec1Pt",  "Elec 1 Pt ",        100, 0, 100);
  hL1EmIso2Pt  = new TH1D("Elec2Pt",  "Elec 2 Pt ",        100, 0, 100);
  hL1EmIso1Eta  = new TH1D("Elec1Eta",  "Elec 1 Eta ",        100, -3, 3);
  hL1EmIso2Eta  = new TH1D("Elec2Eta",  "Elec 2 Eta ",        100, -3, 3);
  
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
  }
  gDirectory->cd();



  gDirectory->cd("/L1Em/NotIsolated/General");
  hL1EmNotIsoMult = new TH1D("ElecMult", "Elec Multiplicity", 10, 0, 10);
  hL1EmNotIso1Pt  = new TH1D("Elec1Pt",  "Elec 1 Pt ",        100, 0, 100);
  hL1EmNotIso2Pt  = new TH1D("Elec2Pt",  "Elec 2 Pt ",        100, 0, 100);
  hL1EmNotIso1Eta  = new TH1D("Elec1Eta",  "Elec 1 Eta ",        100, -3, 3);
  hL1EmNotIso2Eta  = new TH1D("Elec2Eta",  "Elec 2 Eta ",        100, -3, 3);
  
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
  }
  gDirectory->cd();



  //******************
  //Book MET
  //******************
  
  gDirectory->cd("/L1MET/General");
  hL1MET = new TH1D("MET", "MET", 100, 0, 1000);
  gDirectory->cd("/L1MET/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "MET_" + (*l1Names_)[i];
    myHistoTitle = "MET for L1 path " + (*l1Names_)[i];
    hL1METAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
  }

  gDirectory->cd("/L1MET/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "MET_" + (*hlNames_)[i];
    myHistoTitle = "MET for HLT path " + (*hlNames_)[i];    
    hL1METAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
  }
  gDirectory->cd();






  //******************
  //Book histos Reco Objects
  //******************

  //******************
  //Book Jets
  //******************
  
  gDirectory->cd("/RecoJets/General");
  hJetMult = new TH1D("JetMult", "Jet Multiplicity", 30, 0, 30);
  hJet1Pt  = new TH1D("Jet1Pt",  "Jet 1 Pt ",        100, 0, 1000);
  hJet2Pt  = new TH1D("Jet2Pt",  "Jet 2 Pt ",        100, 0, 1000);
  hJet1Eta  = new TH1D("Jet1Eta",  "Jet 1 Eta ",        100, 0, 1000);
  hJet2Eta  = new TH1D("Jet2Eta",  "Jet 2 Eta ",        100, 0, 1000);
  
  gDirectory->cd("/RecoJets/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "JetMult_" + (*l1Names_)[i];
    myHistoTitle = "Jet Multiplicity for L1 path " + (*l1Names_)[i];
    hJetMultAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 20, 0, 20));
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
  }

  gDirectory->cd("/RecoJets/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "JetMult_" + (*hlNames_)[i];
    myHistoTitle = "Jet Multiplicity for HLT path " + (*hlNames_)[i];    
    hJetMultAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 20, 0, 20));
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
  }
  gDirectory->cd();



  //******************
  //Book MET
  //******************
  
  gDirectory->cd("/RecoMET/General");
  hMET = new TH1D("MET", "MET", 100, 0, 1000);
  gDirectory->cd("/RecoMET/L1");
  for(unsigned int i=0; i<l1bits_->size(); i++){
    myHistoName = "MET_" + (*l1Names_)[i];
    myHistoTitle = "MET for L1 path " + (*l1Names_)[i];
    hMETAfterL1.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
  }

  gDirectory->cd("/RecoMET/HLT");
  for(unsigned int i=0; i<hltbits_->size(); i++){
    myHistoName = "MET_" + (*hlNames_)[i];
    myHistoTitle = "MET for HLT path " + (*hlNames_)[i];    
    hMETAfterHLT.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 100, 0, 1000));
  }
  gDirectory->cd();




}


void PlotMaker::handleObjects(const edm::Event& iEvent)
{
  //***********************************************
  // Get the L1 Objects
  //***********************************************

  
  //Get the EM objects
  Handle<l1extra::L1EmParticleCollection> theL1EmIsoHandle, theL1EmNotIsoHandle;
  iEvent.getByLabel(m_l1extra,theL1EmIsoHandle);
  iEvent.getByLabel(m_l1extra,theL1EmNotIsoHandle);
  theL1EmIsoCollection = theL1EmIsoHandle.product();
  theL1EmNotIsoCollection = theL1EmNotIsoHandle.product();

  //Get the Muons  
  Handle<l1extra::L1MuonParticleCollection> theL1MuonHandle;
  iEvent.getByLabel(m_l1extra,theL1MuonHandle);
  theL1MuonCollection = theL1MuonHandle.product();

  //Get the Jets
  Handle<l1extra::L1JetParticleCollection> theL1CentralJetHandle,theL1ForwardJetHandle,theL1TauJetHandle;
  iEvent.getByLabel(m_l1extra,theL1CentralJetHandle);
  iEvent.getByLabel(m_l1extra,theL1ForwardJetHandle);
  iEvent.getByLabel(m_l1extra,theL1TauJetHandle);
  theL1CentralJetCollection = theL1CentralJetHandle.product();
  theL1ForwardJetCollection = theL1ForwardJetHandle.product();
  theL1TauJetCollection = theL1TauJetHandle.product();


  //Get the MET
  Handle<l1extra::L1EtMissParticle> theL1METHandle;
  iEvent.getByLabel(m_l1extra,theL1METHandle);
  theL1METCollection = theL1METHandle.product();


  //***********************************************
  // Get the RECO Objects
  //***********************************************


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
