#include "HLTriggerOffline/JetMET/interface/JetMETDQMPostProcessor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include <iostream>
#include <string.h>
#include <iomanip>
#include<fstream>
#include <math.h>


JetMETDQMPostProcessor::JetMETDQMPostProcessor(const edm::ParameterSet& pset)
{
  subDir_ = pset.getUntrackedParameter<std::string>("subDir");
}

void JetMETDQMPostProcessor::endRun(edm::Run const& run, edm::EventSetup const& es)
{
 //////////////////////////////////
  // setup DQM stor               //
  //////////////////////////////////

  DQMStore * dqm = 0;
  dqm = edm::Service<DQMStore>().operator->();

  if ( ! dqm ) {
    edm::LogInfo("JetMETDQMPostProcessor") << "Cannot create DQMStore instance\n";
    return;
  }


  //go to the directory to be processed
  if(dqm->dirExists(subDir_)) dqm->cd(subDir_);
  else {
   edm::LogWarning("JetMETDQMPostProcessor") << "cannot find directory: " << subDir_ << " , skipping";
    return;
  }
  
  std::vector<std::string> subdirectories = dqm->getSubdirs();
  for(std::vector<std::string>::iterator dir = subdirectories.begin() ;dir!= subdirectories.end(); dir++ ){
    dqm->cd(*dir);
    
    TH1F* gmetTrg = new TH1F("gmetTrg","gmetTrg",100,0,500);
    TH1F* gmetTrg2 = new TH1F("gmetTrg2","gmetTrg2",100,0,500);
    TH1F* gmetTrgLow = new TH1F("gmetTrgLow","gmetTrgLow",100,0,500);
    TH1F* gmet = new TH1F("gmet","gmet",100,0,500);
    TH1F* gjetTrg = new TH1F("gjetTrg","gjetTrg",100,0,500);
    TH1F* gjetTrg2 = new TH1F("gjetTrg2","gjetTrg2",100,0,500);
    TH1F* gjetTrgLow = new TH1F("gjetTrgLow","gjetTrgLow",100,0,500);
    TH1F* gjet = new TH1F("gjet","gjet",100,0,500);
    TH1F* gjetEtaTrg = new TH1F("gjetEtaTrg","gjetEtaTrg",100,-10,10);
    TH1F* gjetEtaTrg2 = new TH1F("gjetEtaTrg2","gjetEtaTrg2",100,-10,10);
    TH1F* gjetEtaTrgLow = new TH1F("gjetEtaTrgLow","gjetEtaTrgLow",100,-10,10);
    TH1F* gjetEta = new TH1F("gjetEta","gjetEta",100,-10,10);
    TH1F* gjetPhiTrg = new TH1F("gjetPhiTrg","gjetPhiTrg",100,-4,4);
    TH1F* gjetPhiTrg2 = new TH1F("gjetPhiTrg2","gjetPhiTrg2",100,-4,4);
    TH1F* gjetPhiTrgLow = new TH1F("gjetPhiTrgLow","gjetPhiTrgLow",100,-4,4);
    TH1F* gjetPhi = new TH1F("gjetPhi","gjetPhi",100,-4,4);
    TH1F* ghtTrg = new TH1F("ghtTrg","ghtTrg",100,0,1000);
    TH1F* ghtTrg2 = new TH1F("ghtTrg2","ghtTrg2",100,0,1000);
    TH1F* ghtTrgLow = new TH1F("ghtTrgLow","ghtTrgLow",100,0,1000);
    TH1F* ght = new TH1F("ght","ght",100,0,1000);
    TH1F* rmetTrg = new TH1F("rmetTrg","rmetTrg",100,0,500);
    TH1F* rmetTrg2 = new TH1F("rmetTrg2","rmetTrg2",100,0,500);
    TH1F* rmetTrgLow = new TH1F("rmetTrgLow","rmetTrgLow",100,0,500);
    TH1F* rmet = new TH1F("rmet","rmet",100,0,500);
    TH1F* rjetTrg = new TH1F("rjetTrg","rjetTrg",100,0,500);
    TH1F* rjetTrg2 = new TH1F("rjetTrg2","rjetTrg2",100,0,500);
    TH1F* rjetTrgLow = new TH1F("rjetTrgLow","rjetTrgLow",100,0,500);
    TH1F* rjet = new TH1F("rjet","rjet",100,0,500);
    TH1F* rjetEtaTrg = new TH1F("rjetEtaTrg","rjetEtaTrg",100,-10,10);
    TH1F* rjetEtaTrg2 = new TH1F("rjetEtaTrg2","rjetEtaTrg2",100,-10,10);
    TH1F* rjetEtaTrgLow = new TH1F("rjetEtaTrgLow","rjetEtaTrgLow",100,-10,10);
    TH1F* rjetEta = new TH1F("rjetEta","rjetEta",100,-10,10);
    TH1F* rjetPhiTrg = new TH1F("rjetPhiTrg","rjetPhiTrg",100,-4,4);
    TH1F* rjetPhiTrg2 = new TH1F("rjetPhiTrg2","rjetPhiTrg2",100,-4,4);
    TH1F* rjetPhiTrgLow = new TH1F("rjetPhiTrgLow","rjetPhiTrgLow",100,-4,4);
    TH1F* rjetPhi = new TH1F("rjetPhi","rjetPhi",100,-4,4);
    TH1F* rhtTrg = new TH1F("rhtTrg","rhtTrg",100,0,1000);
    TH1F* rhtTrg2 = new TH1F("rhtTrg2","rhtTrg2",100,0,1000);
    TH1F* rhtTrgLow = new TH1F("rhtTrgLow","rhtTrgLow",100,0,1000);
    TH1F* rht = new TH1F("rht","rht",100,0,1000);

    TProfile* gmto = new TProfile("_meTurnOngMET","Gen Missing ET Turn-On RelVal",100,0,500,0,1);
    TProfile* gmtol = new TProfile("_meTurnOngMETLow","Gen Missing ET Turn-On Data",100,0,500,0,1);
    TProfile* gjto = new TProfile("_meTurnOngJetPt","Gen Jet Pt Turn-On RelVal",100,0,500,0,1);
    TProfile* gjtol = new TProfile("_meTurnOngJetPtLow","Gen Jet Pt Turn-On Data",100,0,500,0,1);
    TProfile* gjeto = new TProfile("_meTurnOngJetEta","Gen Jet Eta Turn-On RelVal",100,-10,10,0,1);
    TProfile* gjetol = new TProfile("_meTurnOngJetEtaLow","Gen Jet Eta Turn-On Data",100,-10,10,0,1);
    TProfile* gjpto = new TProfile("_meTurnOngJetPhi","Gen Jet Phi Turn-On RelVal",100,-4,4,0,1);
    TProfile* gjptol = new TProfile("_meTurnOngJetPhiLow","Gen Jet Phi Turn-On Data",100,-4,4,0,1);
    TProfile* ghto = new TProfile("_meTurnOngHT","Gen HT Turn-On RelVal",100,0,1000,0,1);
    TProfile* ghtol = new TProfile("_meTurnOngHTLow","Gen HT Turn-On Data",100,0,1000,0,1);
    TProfile* rmto = new TProfile("_meTurnOnrMET","Reco Missing ET Turn-On RelVal",100,0,500,0,1);
    TProfile* rmtol = new TProfile("_meTurnOnrMETLow","Reco Missing ET Turn-On Data",100,0,500,0,1);
    TProfile* rjto = new TProfile("_meTurnOnrJetPt","Reco Jet Pt Turn-On RelVal",100,0,500,0,1);
    TProfile* rjtol = new TProfile("_meTurnOnrJetPtLow","Reco Jet Pt Turn-On Data",100,0,500,0,1);
    TProfile* rjeto = new TProfile("_meTurnOnrJetEta","Reco Jet Eta Turn-On RelVal",100,-10,10,0,1);
    TProfile* rjetol = new TProfile("_meTurnOnrJetEtaLow","Reco Jet Eta Turn-On Data",100,-10,10,0,1);
    TProfile* rjpto = new TProfile("_meTurnOnrJetPhi","Reco Jet Phi Turn-On RelVal",100,-4,4,0,1);
    TProfile* rjptol = new TProfile("_meTurnOnrJetPhiLow","Reco Jet Phi Turn-On Data",100,-4,4,0,1);
    TProfile* rhto = new TProfile("_meTurnOnrHT","Reco HT Turn-On RelVal",100,0,1000,0,1);
    TProfile* rhtol = new TProfile("_meTurnOnrHTLow","Reco HT Turn-On Data",100,0,1000,0,1);

    //std::vector<std::string> mes = dqm->getMEs();
    //for(std::vector<std::string>::iterator me = mes.begin() ;me!= mes.end(); me++ )
    //  std::cout <<*me <<std::endl;
    //std::cout <<std::endl;
    
    gmetTrg->Add(dqm->get(dqm->pwd() + "/_meGenMETTrg")->getTH1F(),1);
    gmetTrg->Sumw2();
    gmetTrg2->Add(dqm->get(dqm->pwd() + "/_meGenMETTrg")->getTH1F(),1);
    gmetTrg2->Sumw2();
    gmetTrgLow->Add(dqm->get(dqm->pwd() + "/_meGenMETTrgLow")->getTH1F(),1);
    gmetTrgLow->Sumw2();
    gmet->Add(dqm->get(dqm->pwd() + "/_meGenMET")->getTH1F(),1);
    gmet->Sumw2();
    gmetTrg->Divide(gmetTrg,gmet,1,1,"B");
    gmetTrg2->Divide(gmetTrg2,gmetTrgLow,1,1,"B");
    
    rmetTrg->Add(dqm->get(dqm->pwd() + "/_meRecoMETTrg")->getTH1F(),1);
    rmetTrg->Sumw2();
    rmetTrg2->Add(dqm->get(dqm->pwd() + "/_meRecoMETTrg")->getTH1F(),1);
    rmetTrg2->Sumw2();
    rmetTrgLow->Add(dqm->get(dqm->pwd() + "/_meRecoMETTrgLow")->getTH1F(),1);
    rmetTrgLow->Sumw2();
    rmet->Add(dqm->get(dqm->pwd() + "/_meRecoMET")->getTH1F(),1);
    rmet->Sumw2();
    rmetTrg->Divide(rmetTrg,rmet,1,1,"B");
    rmetTrg2->Divide(rmetTrg2,rmetTrgLow,1,1,"B");
    
    gjetTrg->Add(dqm->get(dqm->pwd() + "/_meGenJetPtTrg")->getTH1F(),1);
    gjetTrg->Sumw2();
    gjetTrg2->Add(dqm->get(dqm->pwd() + "/_meGenJetPtTrg")->getTH1F(),1);
    gjetTrg2->Sumw2();
    gjetTrgLow->Add(dqm->get(dqm->pwd() + "/_meGenJetPtTrgLow")->getTH1F(),1);
    gjetTrgLow->Sumw2();
    gjet->Add(dqm->get(dqm->pwd() + "/_meGenJetPt")->getTH1F(),1);
    gjet->Sumw2();
    gjetTrg->Divide(gjetTrg,gjet,1,1,"B");
    gjetTrg2->Divide(gjetTrg2,gjetTrgLow,1,1,"B");

    rjetTrg->Add(dqm->get(dqm->pwd() + "/_meRecoJetPtTrg")->getTH1F(),1);
    rjetTrg->Sumw2();
    rjetTrg2->Add(dqm->get(dqm->pwd() + "/_meRecoJetPtTrg")->getTH1F(),1);
    rjetTrg2->Sumw2();
    rjetTrgLow->Add(dqm->get(dqm->pwd() + "/_meRecoJetPtTrgLow")->getTH1F(),1);
    rjetTrgLow->Sumw2();
    rjet->Add(dqm->get(dqm->pwd() + "/_meRecoJetPt")->getTH1F(),1);
    rjet->Sumw2();
    rjetTrg->Divide(rjetTrg,rjet,1,1,"B");
    rjetTrg2->Divide(rjetTrg2,rjetTrgLow,1,1,"B");

    gjetEtaTrg->Add(dqm->get(dqm->pwd() + "/_meGenJetEtaTrg")->getTH1F(),1);
    gjetEtaTrg->Sumw2();
    gjetEtaTrg2->Add(dqm->get(dqm->pwd() + "/_meGenJetEtaTrg")->getTH1F(),1);
    gjetEtaTrg2->Sumw2();
    gjetEtaTrgLow->Add(dqm->get(dqm->pwd() + "/_meGenJetEtaTrgLow")->getTH1F(),1);
    gjetEtaTrgLow->Sumw2();
    gjetEta->Add(dqm->get(dqm->pwd() + "/_meGenJetEta")->getTH1F(),1);
    gjetEta->Sumw2();
    gjetEtaTrg->Divide(gjetEtaTrg,gjetEta,1,1,"B");
    gjetEtaTrg2->Divide(gjetEtaTrg2,gjetEtaTrgLow,1,1,"B");

    rjetEtaTrg->Add(dqm->get(dqm->pwd() + "/_meRecoJetEtaTrg")->getTH1F(),1);
    rjetEtaTrg->Sumw2();
    rjetEtaTrg2->Add(dqm->get(dqm->pwd() + "/_meRecoJetEtaTrg")->getTH1F(),1);
    rjetEtaTrg2->Sumw2();
    rjetEtaTrgLow->Add(dqm->get(dqm->pwd() + "/_meRecoJetEtaTrgLow")->getTH1F(),1);
    rjetEtaTrgLow->Sumw2();
    rjetEta->Add(dqm->get(dqm->pwd() + "/_meRecoJetEta")->getTH1F(),1);
    rjetEta->Sumw2();
    rjetEtaTrg->Divide(rjetEtaTrg,rjetEta,1,1,"B");
    rjetEtaTrg2->Divide(rjetEtaTrg2,rjetEtaTrgLow,1,1,"B");
    
    gjetPhiTrg->Add(dqm->get(dqm->pwd() + "/_meGenJetPhiTrg")->getTH1F(),1);
    gjetPhiTrg->Sumw2();
    gjetPhiTrg2->Add(dqm->get(dqm->pwd() + "/_meGenJetPhiTrg")->getTH1F(),1);
    gjetPhiTrg2->Sumw2();
    gjetPhiTrgLow->Add(dqm->get(dqm->pwd() + "/_meGenJetPhiTrgLow")->getTH1F(),1);
    gjetPhiTrgLow->Sumw2();
    gjetPhi->Add(dqm->get(dqm->pwd() + "/_meGenJetPhi")->getTH1F(),1);
    gjetPhi->Sumw2();
    gjetPhiTrg->Divide(gjetPhiTrg,gjetPhi,1,1,"B");
    gjetPhiTrg2->Divide(gjetPhiTrg2,gjetPhiTrgLow,1,1,"B");
    
    rjetPhiTrg->Add(dqm->get(dqm->pwd() + "/_meRecoJetPhiTrg")->getTH1F(),1);
    rjetPhiTrg->Sumw2();
    rjetPhiTrg2->Add(dqm->get(dqm->pwd() + "/_meRecoJetPhiTrg")->getTH1F(),1);
    rjetPhiTrg2->Sumw2();
    rjetPhiTrgLow->Add(dqm->get(dqm->pwd() + "/_meRecoJetPhiTrgLow")->getTH1F(),1);
    rjetPhiTrgLow->Sumw2();
    rjetPhi->Add(dqm->get(dqm->pwd() + "/_meRecoJetPhi")->getTH1F(),1);
    rjetPhi->Sumw2();
    rjetPhiTrg->Divide(rjetPhiTrg,rjetPhi,1,1,"B");
    rjetPhiTrg2->Divide(rjetPhiTrg2,rjetPhiTrgLow,1,1,"B");
    
    ghtTrg->Add(dqm->get(dqm->pwd() + "/_meGenHTTrg")->getTH1F(),1);
    ghtTrg->Sumw2();
    ghtTrg2->Add(dqm->get(dqm->pwd() + "/_meGenHTTrg")->getTH1F(),1);
    ghtTrg2->Sumw2();
    ghtTrgLow->Add(dqm->get(dqm->pwd() + "/_meGenHTTrgLow")->getTH1F(),1);
    ghtTrgLow->Sumw2();
    ght->Add(dqm->get(dqm->pwd() + "/_meGenHT")->getTH1F(),1);
    ght->Sumw2();
    ghtTrg->Divide(ghtTrg,ght,1,1,"B");
    ghtTrg2->Divide(ghtTrg2,ghtTrgLow,1,1,"B");

    rhtTrg->Add(dqm->get(dqm->pwd() + "/_meRecoHTTrg")->getTH1F(),1);
    rhtTrg->Sumw2();
    rhtTrg2->Add(dqm->get(dqm->pwd() + "/_meRecoHTTrg")->getTH1F(),1);
    rhtTrg2->Sumw2();
    rhtTrgLow->Add(dqm->get(dqm->pwd() + "/_meRecoHTTrgLow")->getTH1F(),1);
    rhtTrgLow->Sumw2();
    rht->Add(dqm->get(dqm->pwd() + "/_meRecoHT")->getTH1F(),1);
    rht->Sumw2();
    rhtTrg->Divide(rhtTrg,rht,1,1,"B");
    rhtTrg2->Divide(rhtTrg2,rhtTrgLow,1,1,"B");
    
    double val,err;
    for (int ib=0;ib<100;ib++) {
      //genmet relval
      val = gmetTrg->GetBinContent(ib+1);
      gmto->SetBinContent(ib+1,val);
      gmto->SetBinEntries(ib+1,1);
      err = gmetTrg->GetBinError(ib+1);
      gmto->SetBinError(ib+1,sqrt(err*err+val*val));
      //genmet data
      val = gmetTrg2->GetBinContent(ib+1);
      gmtol->SetBinContent(ib+1,val);
      gmtol->SetBinEntries(ib+1,1);
      err = gmetTrg2->GetBinError(ib+1);
      gmtol->SetBinError(ib+1,sqrt(err*err+val*val));
      //recmet relval
      val = rmetTrg->GetBinContent(ib+1);
      rmto->SetBinContent(ib+1,val);
      rmto->SetBinEntries(ib+1,1);
      err = rmetTrg->GetBinError(ib+1);
      rmto->SetBinError(ib+1,sqrt(err*err+val*val));
      //recmet data
      val = rmetTrg2->GetBinContent(ib+1);
      rmtol->SetBinContent(ib+1,val);
      rmtol->SetBinEntries(ib+1,1);
      err = rmetTrg2->GetBinError(ib+1);
      rmtol->SetBinError(ib+1,sqrt(err*err+val*val));
      //genjet relval
      val = gjetTrg->GetBinContent(ib+1);
      gjto->SetBinContent(ib+1,val);
      gjto->SetBinEntries(ib+1,1);
      err = gjetTrg->GetBinError(ib+1);
      gjto->SetBinError(ib+1,sqrt(err*err+val*val));
      //genjet data
      val = gjetTrg2->GetBinContent(ib+1);
      gjtol->SetBinContent(ib+1,val);
      gjtol->SetBinEntries(ib+1,1);
      err = gjetTrg2->GetBinError(ib+1);
      gjtol->SetBinError(ib+1,sqrt(err*err+val*val));
      //recjet relval
      val = rjetTrg->GetBinContent(ib+1);
      rjto->SetBinContent(ib+1,val);
      rjto->SetBinEntries(ib+1,1);
      err = rjetTrg->GetBinError(ib+1);
      rjto->SetBinError(ib+1,sqrt(err*err+val*val));
      //recjet data
      val = rjetTrg2->GetBinContent(ib+1);
      rjtol->SetBinContent(ib+1,val);
      rjtol->SetBinEntries(ib+1,1);
      err = rjetTrg2->GetBinError(ib+1);
      rjtol->SetBinError(ib+1,sqrt(err*err+val*val));
      //genjeteta relval
      val = gjetEtaTrg->GetBinContent(ib+1);
      gjeto->SetBinContent(ib+1,val);
      gjeto->SetBinEntries(ib+1,1);
      err = gjetEtaTrg->GetBinError(ib+1);
      gjeto->SetBinError(ib+1,sqrt(err*err+val*val));
      //genjeteta data
      val = gjetEtaTrg2->GetBinContent(ib+1);
      gjetol->SetBinContent(ib+1,val);
      gjetol->SetBinEntries(ib+1,1);
      err = gjetEtaTrg2->GetBinError(ib+1);
      gjetol->SetBinError(ib+1,sqrt(err*err+val*val));
      //recjeteta relval
      val = rjetEtaTrg->GetBinContent(ib+1);
      rjeto->SetBinContent(ib+1,val);
      rjeto->SetBinEntries(ib+1,1);
      err = rjetEtaTrg->GetBinError(ib+1);
      rjeto->SetBinError(ib+1,sqrt(err*err+val*val));
      //recjeteta data
      val = rjetEtaTrg2->GetBinContent(ib+1);
      rjetol->SetBinContent(ib+1,val);
      rjetol->SetBinEntries(ib+1,1);
      err = rjetEtaTrg2->GetBinError(ib+1);
      rjetol->SetBinError(ib+1,sqrt(err*err+val*val));
      //genjetphi relval
      val = gjetPhiTrg->GetBinContent(ib+1);
      gjpto->SetBinContent(ib+1,val);
      gjpto->SetBinEntries(ib+1,1);
      err = gjetPhiTrg->GetBinError(ib+1);
      gjpto->SetBinError(ib+1,sqrt(err*err+val*val));
      //genjetphi data
      val = gjetPhiTrg2->GetBinContent(ib+1);
      gjptol->SetBinContent(ib+1,val);
      gjptol->SetBinEntries(ib+1,1);
      err = gjetPhiTrg2->GetBinError(ib+1);
      gjptol->SetBinError(ib+1,sqrt(err*err+val*val));
      //recjetphi relval
      val = rjetPhiTrg->GetBinContent(ib+1);
      rjpto->SetBinContent(ib+1,val);
      rjpto->SetBinEntries(ib+1,1);
      err = rjetPhiTrg->GetBinError(ib+1);
      rjpto->SetBinError(ib+1,sqrt(err*err+val*val));
      //recjetphi data
      val = rjetPhiTrg2->GetBinContent(ib+1);
      rjptol->SetBinContent(ib+1,val);
      rjptol->SetBinEntries(ib+1,1);
      err = rjetPhiTrg2->GetBinError(ib+1);
      rjptol->SetBinError(ib+1,sqrt(err*err+val*val));
      //genht relval
      val = ghtTrg->GetBinContent(ib+1);
      ghto->SetBinContent(ib+1,val);
      ghto->SetBinEntries(ib+1,1);
      err = ghtTrg->GetBinError(ib+1);
      ghto->SetBinError(ib+1,sqrt(err*err+val*val));
      //genht data
      val = ghtTrg2->GetBinContent(ib+1);
      ghtol->SetBinContent(ib+1,val);
      ghtol->SetBinEntries(ib+1,1);
      err = ghtTrg2->GetBinError(ib+1);
      ghtol->SetBinError(ib+1,sqrt(err*err+val*val));
      //recht relval
      val = rhtTrg->GetBinContent(ib+1);
      rhto->SetBinContent(ib+1,val);
      rhto->SetBinEntries(ib+1,1);
      err = rhtTrg->GetBinError(ib+1);
      rhto->SetBinError(ib+1,sqrt(err*err+val*val));
      //recht data
      val = rhtTrg2->GetBinContent(ib+1);
      rhtol->SetBinContent(ib+1,val);
      rhtol->SetBinEntries(ib+1,1);
      err = rhtTrg2->GetBinError(ib+1);
      rhtol->SetBinError(ib+1,sqrt(err*err+val*val));


      //std::cout <<"MET:"<<_meTurnOnMET->getBinContent(ib+1)<<" "<<gmetTrg->GetBinContent(ib+1)<<" "<<_meTurnOnMET->getBinError(ib+1)<<" "<<gmetTrg->GetBinError(ib+1)<<std::endl;
      //std::cout <<"JET:"<<_meTurnOnJetPt->getBinContent(ib+1)<<" "<<gjetTrg->GetBinContent(ib+1)<<" "<<_meTurnOnJetPt->getBinError(ib+1)<<" "<<gjetTrg->GetBinError(ib+1)<<std::endl;
    }
    dqm->bookProfile("Gen Missing ET Turn-On RelVal",gmto);
    dqm->bookProfile("Gen Missing ET Turn-On Data",gmtol);
    dqm->bookProfile("Reco Missing ET Turn-On RelVal",rmto);
    dqm->bookProfile("Reco Missing ET Turn-On Data",rmtol);
    dqm->bookProfile("Gen Jet Pt Turn-On RelVal",gjto);
    dqm->bookProfile("Gen Jet Pt Turn-On Data",gjtol);
    dqm->bookProfile("Reco Jet Pt Turn-On RelVal",rjto);
    dqm->bookProfile("Reco Jet Pt Turn-On Data",rjtol);
    dqm->bookProfile("Gen Jet Eta Turn-On RelVal",gjeto);
    dqm->bookProfile("Gen Jet Eta Turn-On Data",gjetol);
    dqm->bookProfile("Reco Jet Eta Turn-On RelVal",rjeto);
    dqm->bookProfile("Reco Jet Eta Turn-On Data",rjetol);
    dqm->bookProfile("Gen Jet Phi Turn-On RelVal",gjpto);
    dqm->bookProfile("Gen Jet Phi Turn-On Data",gjptol);
    dqm->bookProfile("Reco Jet Phi Turn-On RelVal",rjpto);
    dqm->bookProfile("Reco Jet Phi Turn-On Data",rjptol);
    dqm->bookProfile("Gen HT Turn-On RelVal",ghto);
    dqm->bookProfile("Gen HT Turn-On Data",ghtol);
    dqm->bookProfile("Reco HT Turn-On RelVal",rhto);
    dqm->bookProfile("Reco HT Turn-On Data",rhtol);
    delete gjto;
    delete gjtol;
    delete gjeto;
    delete gjetol;
    delete gjpto;
    delete gjptol;
    delete rjto;
    delete rjtol;
    delete rjeto;
    delete rjetol;
    delete rjpto;
    delete rjptol;
    delete gmto;
    delete gmtol;
    delete rmto;
    delete rmtol;
    delete ghto;
    delete ghtol;
    delete rhto;
    delete rhtol;
    delete gmetTrg;
    delete gmetTrg2;
    delete gmetTrgLow;
    delete gmet;
    delete rmetTrg;
    delete rmetTrg2;
    delete rmetTrgLow;
    delete rmet;
    delete gjetTrg;
    delete gjetTrg2;
    delete gjetTrgLow;
    delete gjet;
    delete rjetTrg;
    delete rjetTrg2;
    delete rjetTrgLow;
    delete rjet;
    delete gjetEtaTrg;
    delete gjetEtaTrg2;
    delete gjetEtaTrgLow;
    delete gjetEta;
    delete rjetEtaTrg;
    delete rjetEtaTrg2;
    delete rjetEtaTrgLow;
    delete rjetEta;
    delete gjetPhiTrg;
    delete gjetPhiTrg2;
    delete gjetPhiTrgLow;
    delete gjetPhi;
    delete rjetPhiTrg;
    delete rjetPhiTrg2;
    delete rjetPhiTrgLow;
    delete rjetPhi;
    delete ghtTrg;
    delete ghtTrg2;
    delete ghtTrgLow;
    delete ght;
    delete rhtTrg;
    delete rhtTrg2;
    delete rhtTrgLow;
    delete rht;

    dqm->goUp(); 
  }
}
void JetMETDQMPostProcessor::endJob()
{
  /*
  //////////////////////////////////
  // setup DQM stor               //
  //////////////////////////////////

  DQMStore * dqm = 0;
  dqm = edm::Service<DQMStore>().operator->();

  if ( ! dqm ) {
    edm::LogInfo("JetMETDQMPostProcessor") << "Cannot create DQMStore instance\n";
    return;
  }


  //go to the directory to be processed
  if(dqm->dirExists(subDir_)) dqm->cd(subDir_);
  else {
   edm::LogWarning("JetMETDQMPostProcessor") << "cannot find directory: " << subDir_ << " , skipping";
    return;
  }

  std::vector<std::string> subdirectories = dqm->getSubdirs();
  for(std::vector<std::string>::iterator dir = subdirectories.begin() ;dir!= subdirectories.end(); dir++ ){
    dqm->cd(*dir);
    
    MonitorElement*_meTurnOnMET = dqm->book1D("_meTurnOnMET","Missing ET Turn-On",100,0,500);
    MonitorElement*_meTurnOnJetPt = dqm->book1D("_meTurnOnJetPt","Jet Pt Turn-On",100,0,500);
    
    //std::vector<std::string> mes = dqm->getMEs();
    //for(std::vector<std::string>::iterator me = mes.begin() ;me!= mes.end(); me++ )
    //  std::cout <<*me <<std::endl;
    //std::cout <<std::endl;
    
    _meTurnOnMET->getTH1F()->Add(dqm->get(dqm->pwd() + "/_meGenMETTrg")->getTH1F(),1);
    _meTurnOnMET->getTH1F()->Sumw2();
    dqm->get(dqm->pwd() + "/_meGenMET")->getTH1F()->Sumw2();
    _meTurnOnMET->getTH1F()->Divide(_meTurnOnMET->getTH1F(),dqm->get(dqm->pwd() + "/_meGenMET")->getTH1F(),1,1,"B");
    
    _meTurnOnJetPt->getTH1F()->Add(dqm->get(dqm->pwd() + "/_meGenJetPtTrg")->getTH1F(),1);
    _meTurnOnJetPt->getTH1F()->Sumw2();
    dqm->get(dqm->pwd() + "/_meGenJetPt")->getTH1F()->Sumw2();
    _meTurnOnJetPt->getTH1F()->Divide(_meTurnOnJetPt->getTH1F(),dqm->get(dqm->pwd() + "/_meGenJetPt")->getTH1F(),1,1,"B");
    
    dqm->goUp();
  }
  */
}
DEFINE_FWK_MODULE(JetMETDQMPostProcessor);
