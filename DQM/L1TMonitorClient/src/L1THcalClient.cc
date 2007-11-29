#include "DQM/L1TMonitorClient/interface/L1THcalClient.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/ClientConfig/interface/SubscriptionHandle.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include <DQMServices/UI/interface/MonitorUIRoot.h>
#include "DQMServices/CoreROOT/interface/MonitorElementRootT.h"
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TPaveStats.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TF1.h>
#include <TRandom.h>
using namespace edm;
using namespace std;

// Local definitions for the limits of the histograms
const unsigned int RTPBINS = 101;
const float RTPMIN = -0.5;
const float RTPMAX = 100.5;

const unsigned int TPPHIBINS = 72;
const float TPPHIMIN = 0.5;
const float TPPHIMAX = 72.5;

const unsigned int TPETABINS = 65;
const float TPETAMIN = -32.5;
const float TPETAMAX = 32.5;

const unsigned int effBins = 50;
const float effMinHBHE = -0.5;
const float effMaxHBHE = 5.5;
const float effMinHF = -0.5;
const float effMaxHF = 2.5;

const unsigned int ratiobins = 100;
const float ratiomin = 0.0;
const float ratiomax = 1.0;

const unsigned int tvsrecbins = 100;
const float tvsrecmin = 0.0;
const float tvsrecmax = 100.0;

const unsigned int effcombo = 6472;
const float effcombomin = -3272;
const float effcombomax = 3272;

L1THcalClient::L1THcalClient(const edm::ParameterSet& iConfig): L1TBaseClient()
{
  saveOutput = iConfig.getUntrackedParameter<bool>("saveOutput", false);
  outputFile = iConfig.getUntrackedParameter<string>("outputFile", "L1THcalClient.root");
  stdalone = iConfig.getUntrackedParameter<bool>("Standalone",false);
  minEventsforFit = iConfig.getUntrackedParameter<int>("minEventsforFit",1000);
  input_dir = "L1TMonitor/L1THCALTPGXAna/";
  output_dir = "L1TMonitor/QTests/";

  //qualityCriterionName = iConfig.getUntrackedParameter<string>("qualityTestName","testYRange");
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->showDirStructure();
  dbe->setVerbose(1); 

  if(stdalone){ 
  getMESubscriptionListFromFile = iConfig.getUntrackedParameter<bool>("getMESubscriptionListFromFile", true);
  getQualityTestsFromFile = iConfig.getUntrackedParameter<bool>("getQualityTestsFromFile", true);

  subscriber=new SubscriptionHandle;
  qtHandler=new QTestHandle;

  if (getMESubscriptionListFromFile)
  subscriber->getMEList("MESubscriptionList.xml"); 
  if (getQualityTestsFromFile)
  qtHandler->configureTests("QualityTests.xml",mui_->getBEInterface());

  }
  
  LogInfo( "TriggerDQM");
}

// ---------------------------------------------------------

L1THcalClient::~L1THcalClient()
{
 
 LogInfo("TriggerDQM")<<"[TriggerDQM]: ending... ";

}


// ------------ method called to for each event  ------------

void
L1THcalClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

// access to Geometry if needed
  nevents++;
  if (!stdalone || (nevents%10 == 0))    LogInfo("TriggerDQM")<<"[TriggerDQM]: event analyzed "<<nevents;

}

// ------------ method called once each job just before starting event loop  ------------

void L1THcalClient::beginJob(const edm::EventSetup&)
{
  LogInfo("TriggerDQM")<<"[TriggerDQM]: Begin Job";
  LogInfo("TriggerDQM")<<"[TriggerDQM]: Standalone = "<<stdalone;
  nevents = 0;
  dbe->setCurrentFolder("L1TMonitor/QTests");
  //2-D plots
  hcalPlateau_ =
    dbe->book2D("FitPlateau","HCAL Fit Plateau",TPETABINS,TPETAMIN,TPETAMAX,
		TPPHIBINS,TPPHIMIN,TPPHIMAX);
  hcalThreshold_ =
    dbe->book2D("FitThreshold","HCAL Fit Threshold",TPETABINS,TPETAMIN,TPETAMAX,
                TPPHIBINS,TPPHIMIN,TPPHIMAX);
  hcalWidth_ =
    dbe->book2D("FitWidth","HCAL Fit Width",TPETABINS,TPETAMIN,TPETAMAX,
                TPPHIBINS,TPPHIMIN,TPPHIMAX);

  //1-D plots

  if(stdalone){
   subscriber->makeSubscriptions(mui_);
   qtHandler->attachTests(mui_->getBEInterface());	
  }

}

// ------------ method called once each job just after ending the event loop  ------------
void L1THcalClient::endJob() {

   LogInfo("TriggerDQM")<<"[TriggerDQM]: endJob";

}

void L1THcalClient::endLuminosityBlock(const edm::LuminosityBlock & iLumiSection, const edm::EventSetup & iSetup) {

  LogInfo("TriggerDQM")<<"[TriggerDQM]: end Lumi Section.";
  dbe->setCurrentFolder("L1TMonitor/QTests");
  TF1 *turnon = new TF1("turnon","[0]*0.5*(TMath::Erf((x -[1])*0.5/[2])+1.)",0,30);
  TH1F *eff_histo;
  //int ilumi = iLumiSection.id().luminosityBlock();
  // if(stdalone) mui_->doMonitoring();
  // ----------------- QT examples example
  // ----------------- get bin content and create new ME, then perform QT

  //efficiency by region
  //std::cout << "before eff calc \n";
  TH1F *hcalEff_num = this->get1DHisto(input_dir+"HcalTP1",dbe);
  if (!hcalEff_num) std::cout << "numerator not found\n";
  TH1F *hcalEff_den = this->get1DHisto(input_dir+"HcalAll1",dbe);
  if (!hcalEff_den) std::cout << "denominator not found\n";
  hcalEff_1_ =
    dbe->book1D("HcalEff1","HCAL Efficiency - 1",effBins,effMinHBHE,effMaxHBHE);
  calcEff(hcalEff_num, hcalEff_den, hcalEff_1_);
  if (hcalEff_num->GetEntries() > minEventsforFit)
    {
      eff_histo = this->get1DHisto(output_dir+"HcalEff1",dbe);
      turnon->SetParameter(0,1);
      turnon->SetParameter(1,2);
      turnon->SetParameter(2,6);
      eff_histo->Fit("turnon","LL,E");
    }
  hcalEff_num = this->get1DHisto(input_dir+"HcalTP2",dbe);
  if (!hcalEff_num) std::cout << "numerator not found\n";
  hcalEff_den = this->get1DHisto(input_dir+"HcalAll2",dbe);
  if (!hcalEff_den) std::cout << "denominator not found\n";
  hcalEff_2_ =
    dbe->book1D("HcalEff2","HCAL Efficiency - 2",effBins,effMinHBHE,effMaxHBHE);
  calcEff(hcalEff_num, hcalEff_den, hcalEff_2_);
    if (hcalEff_num->GetEntries() > minEventsforFit)
      {
	eff_histo = this->get1DHisto(output_dir+"HcalEff2",dbe);
	turnon->SetParameter(0,1);
	turnon->SetParameter(1,2);
	turnon->SetParameter(2,6);
	eff_histo->Fit("turnon","LL,E");
      }
  hcalEff_num = this->get1DHisto(input_dir+"HcalTP3",dbe);
  if (!hcalEff_num) std::cout << "numerator not found\n";
  hcalEff_den = this->get1DHisto(input_dir+"HcalAll3",dbe);
  if (!hcalEff_den) std::cout << "denominator not found\n";
  hcalEff_3_ =
    dbe->book1D("HcalEff3","HCAL Efficiency - 3",effBins,effMinHBHE,effMaxHBHE);
  calcEff(hcalEff_num, hcalEff_den, hcalEff_3_);
  if (hcalEff_num->GetEntries() > minEventsforFit)
    {
      eff_histo = this->get1DHisto(output_dir+"HcalEff3",dbe);
      turnon->SetParameter(0,1);
      turnon->SetParameter(1,3);
      turnon->SetParameter(2,6);
      eff_histo->Fit("turnon","LL,E");
    }
  hcalEff_num = this->get1DHisto(input_dir+"HcalTP4",dbe);
  if (!hcalEff_num) std::cout << "numerator not found\n";
  hcalEff_den = this->get1DHisto(input_dir+"HcalAll4",dbe);
  if (!hcalEff_den) std::cout << "denominator not found\n";
  hcalEff_4_ =
    dbe->book1D("HcalEff4","HCAL Efficiency - 4",effBins,effMinHF,effMaxHF);
  calcEff(hcalEff_num, hcalEff_den, hcalEff_4_);
  if (hcalEff_num->GetEntries() > minEventsforFit)
    {
      eff_histo = this->get1DHisto(output_dir+"HcalEff4",dbe);
      turnon->SetParameter(0,1);
      turnon->SetParameter(1,1);
      turnon->SetParameter(2,6);
      eff_histo->Fit("turnon","LL,E");
    }
  double plateau, threshold, width;

  //efficiency histos for HBHE
  for (int i=0; i < 56; i++)
    {
      char hname[20],htitle[30];
      int ieta, iphi;
      for (int j=0; j < 72; j++)
	{
	  iphi = j+1;
	  if (i<28) ieta = i-28;
	  else ieta = i-27;
	  sprintf(hname,"eff_%d_%d",ieta,iphi);
	  sprintf(htitle,"Eff  <%d,%d>",ieta,iphi);
	  hcalEff_HBHE[i][j] = dbe->book1D(hname, htitle, effBins,effMinHBHE,effMaxHBHE);
	  hcalEff_num = this->get1DHisto(input_dir+(string)hname+"_num",dbe);
	  hcalEff_den = this->get1DHisto(input_dir+(string)hname+"_den",dbe);
	  if (!hcalEff_num) std::cout << "numerator not found\n";
	  if (!hcalEff_den) std::cout << "denominator not found\n";
	  calcEff(hcalEff_num, hcalEff_den, hcalEff_HBHE[i][j]);

	  if (hcalEff_num->GetEntries() > minEventsforFit)
	    {
	      eff_histo = this->get1DHisto(output_dir+(string)hname,dbe);
	      turnon->SetParameter(0,1);
	      turnon->SetParameter(1,3);
	      turnon->SetParameter(2,6);
	      eff_histo->Fit("turnon","LL,E");
	      plateau = turnon->GetParameter(0);
	      threshold = turnon->GetParameter(1);
	      width = turnon->GetParameter(2);
	      hcalPlateau_->Fill(ieta,iphi,plateau);
	      hcalThreshold_->Fill(ieta,iphi,threshold);
	      hcalWidth_->Fill(ieta,iphi,width);
	    }
	}
    }
  //efficiency histos for HF
  for (int i=0; i < 8; i++)
    {
      char hname[20],htitle[30];
      int ieta, iphi;
      for (int j=0; j < 18; j++)
	{
	  iphi = j*4+1;
	  if (i<4) ieta = i-32;
	  else ieta = i+25;
	  sprintf(hname,"eff_%d_%d",ieta,iphi);
	  sprintf(htitle,"Eff  <%d,%d>",ieta,iphi);
	  hcalEff_HF[i][j] = dbe->book1D(hname, htitle, effBins,effMinHF,effMaxHF);
          hcalEff_num = this->get1DHisto(input_dir+(string)hname+"_num",dbe);
          hcalEff_den = this->get1DHisto(input_dir+(string)hname+"_den",dbe);
          if (!hcalEff_num) std::cout << "numerator not found\n";
          if (!hcalEff_den) std::cout << "denominator not found\n";
          calcEff(hcalEff_num, hcalEff_den, hcalEff_HF[i][j]);
	  if (hcalEff_num->GetEntries() > minEventsforFit)
	    {
	      eff_histo = this->get1DHisto(output_dir+(string)hname,dbe);
	      turnon->SetParameter(0,1);
	      turnon->SetParameter(1,1);
	      turnon->SetParameter(2,6);
	      eff_histo->Fit("turnon","LL,E");
	      plateau = turnon->GetParameter(0);
	      threshold = turnon->GetParameter(1);
	      width = turnon->GetParameter(2);
	      hcalPlateau_->Fill(ieta,iphi,plateau);
	      hcalThreshold_->Fill(ieta,iphi,threshold);
	      hcalWidth_->Fill(ieta,iphi,width);
	    }
	}
    }

  // ----------------- save results
  //add tests here --> one for each final plot
  //if(stdalone){
  //   mui_->getBEInterface()->runQTests();
  //   qtHandler->checkGlobalQTStatus(mui_->getBEInterface());
  // }
  // this->getReport("L1TMonitor/QTests/normGTFEBx", dbe, qualityCriterionName);
  dbe->save(outputFile);
}

void L1THcalClient::calcEff(TH1F *num, TH1F *den, MonitorElement* me)
{
  if (num->GetNbinsX() != den->GetNbinsX()) 
    {
      std::cout << "numerator and denominator do not have the same number of bins!\n";
      return;
    }
  double eff;
  for (int bin = 0; bin <= num->GetNbinsX(); bin++)
    {
      if (den->GetBinContent(bin) != 0) eff = num->GetBinContent(bin)/den->GetBinContent(bin);
      else eff = 0;
      me->setBinContent(bin,eff);
    }
}

