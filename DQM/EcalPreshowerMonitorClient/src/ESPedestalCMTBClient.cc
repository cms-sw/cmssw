#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalCMTBClient.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESDQMUtils.h"

#include "TStyle.h"
#include "TH2F.h"
#include "TPaveText.h"

ESPedestalCMTBClient::ESPedestalCMTBClient(const ParameterSet& ps) {
  
  writeHisto_     = ps.getUntrackedParameter<bool>("writeHisto", true);
  writeHTML_      = ps.getUntrackedParameter<bool>("writeHTML", true);
  dumpRate_       = ps.getUntrackedParameter<int>("dumpRate", 100);
  outputFileName_ = ps.getUntrackedParameter<string>("outputFileName", "ESPedestalCMTB");
  rootFolder_     = ps.getUntrackedParameter<string>("rootFolder", "");
  htmlDir_        = ps.getUntrackedParameter<string>("htmlDir","/preshower/DQM/CT");
  htmlName_       = ps.getUntrackedParameter<string>("htmlName","ESPedestalCMTB.html");  
  cmnThreshold_   = ps.getUntrackedParameter<double>("cmnThreshold", 3);
  sta_            = ps.getUntrackedParameter<bool>("RunStandalone", false);

  count_ = 0;
  run_ = -1;
  init_ = false;

  dbe_ = Service<DaqMonitorBEInterface>().operator->();
}

ESPedestalCMTBClient::~ESPedestalCMTBClient(){
}

void ESPedestalCMTBClient::beginJob(const EventSetup& context){

  if (dbe_) {
    dbe_->setVerbose(1);
    dbe_->setCurrentFolder("ES/QT/PedestalCMTB");
    dbe_->rmdir("ES/QT/PedestalCMTB");
  }

}

void ESPedestalCMTBClient::endJob(){
  
  Char_t runNum_s[50];
  sprintf(runNum_s, "%08d", run_);
  outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
  
  if (writeHTML_) {
    doQT();
    htmlOutput(run_, htmlDir_, htmlName_);
  }

  if (writeHisto_) dbe_->save(outputFile_);
  dbe_->rmdir("ES/QT/PedestalCMTB");  

  if ( init_ ) this->cleanup();
}

void ESPedestalCMTBClient::setup() {

  init_ = true;

  Char_t hist[200];
  
  if (dbe_) {
    dbe_->setCurrentFolder("ES/QT/PedestalCMTB");
    sprintf(hist, "ES QT CMTB Mean");
    meMean_ = dbe_->book1D(hist, hist, 200, -100, 100);
    sprintf(hist, "ES QT CMTB RMS");
    meRMS_ = dbe_->book1D(hist, hist, 200, -100, 100);

    for (int i=0; i<2; ++i) {
	sprintf(hist, "ES CMTB Quality Plane %d", i+1);
	meCMCol_[i] = dbe_->book2D(hist, hist, 4, 0, 4, 4, 0, 4);
    }
  }
  
}

void ESPedestalCMTBClient::cleanup() {

  if (sta_) return;

  if (dbe_) {
    dbe_->setCurrentFolder("ES/QT/PedestalCMTB");
    if (meMean_) dbe_->removeElement( meMean_->getName() );
    if (meRMS_) dbe_->removeElement( meRMS_->getName() );
    meMean_ = 0;
    meRMS_ = 0;
    for (int i=0; i<2; ++i) {
      if (meCMCol_[i]) dbe_->removeElement( meCMCol_[i]->getName() );
      meCMCol_[i] = 0;
    }
  }
  
  init_ = false;
}

void ESPedestalCMTBClient::analyze(const Event& e, const EventSetup& context){
  
  if ( ! init_ ) this->setup();

  int runNum = e.id().run();

  if (runNum != run_) { 

    if (run_ > 0) {
      Char_t runNum_s[50];
      sprintf(runNum_s, "%08d", run_);
      outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
      
      if (writeHTML_) {
	doQT();
	htmlOutput(run_, htmlDir_, htmlName_);
      }

      if (writeHisto_) dbe_->save(outputFile_);
    }

    run_ = runNum; 
    count_ = 0;
  }

  count_++;

  if ((count_ % dumpRate_) == 0) {
    if (writeHTML_) {
      doQT();
      htmlOutput(runNum, htmlDir_, htmlName_);
    }
    if (writeHisto_) dbe_->save(outputFile_);
  }
  
}

void ESPedestalCMTBClient::doQT() {

  ESDQMUtils::resetME( meMean_ );
  ESDQMUtils::resetME( meRMS_ );

  int val = 0;
  for (int i=0; i<2; ++i) {    
    for (int j=30; j<34; ++j) {
      for (int k=19; k<23; ++k) {
	
	MonitorElement * senME = dbe_->get(getMEName(i+1, j, k, 0, 1));
	
	if (senME) {
	  MonitorElementT<TNamed>* sen = dynamic_cast<MonitorElementT<TNamed>*>(senME);           
	  TH1F *hCMSen = dynamic_cast<TH1F*> (sen->operator->());  	    
	  
	  if (hCMSen->GetRMS()>cmnThreshold_) val = 7;
	  else if (hCMSen->GetEntries() == 0) val = 5;
	  else val = 4;
	  
	  meCMCol_[i]->setBinContent(j-29, k-18, val) ;  
	    
	  if (hCMSen->GetEntries() != 0) {
	    meMean_->Fill(hCMSen->GetMean());
	    meRMS_->Fill(hCMSen->GetRMS());
	  }
	  
	}
	
      }	
    }
  }
  
}

string ESPedestalCMTBClient::getMEName(const int & plane, const int & row, const int & col, const int & strip, const int & type) {
  
  Char_t hist[500];
  if (type == 0)
    sprintf(hist,"%sES/ESpedestalCMTBTask/ES Pedestal P %d Row %02d Col %02d Str %02d", rootFolder_.c_str(),plane,row,col,strip);
  else 
    sprintf(hist,"%sES/ESPedestalCMTBTask/ES Sensor CM P %d Row %02d Col %02d", rootFolder_.c_str(),plane,row,col);

  return hist;
}

void ESPedestalCMTBClient::htmlOutput(int run, string htmlDir, string htmlName) {

  cout<<"Going to output ESPedestalCMTBClient html ..."<<endl;
  
  Char_t run_s[50];
  sprintf(run_s, "%08d", run); 
  htmlDir = htmlDir+"/"+run_s;
  system(("/bin/mkdir -m 777 -p " + htmlDir).c_str());

  ofstream htmlFile;   
  htmlFile.open((htmlDir+"/"+htmlName).c_str()); 

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=UTF-8\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Preshower DQM : PedestalCMTBTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run Number / Num of Analyzed events :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;"<< count_ <<"</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task :&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Test Beam Common Mode</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>This strip has problems</td>" << endl;
  htmlFile << "<td bgcolor=lime>This strip has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>This strip is missing</td></table>" << endl;
  htmlFile << "<br>" << endl;

  // make plots
  string histName;
  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1, 0);
  gStyle->SetStatW(0.3);
  gStyle->SetStatH(0.3);
  gStyle->SetGridStyle(1);

  TCanvas *cCMQ = new TCanvas("cCMQ", "cCMQ", 600, 300);
  TCanvas *cCM  = new TCanvas("cCM",  "cCM",  600, 300);
  
  MonitorElementT<TNamed>* CMQ[2];
  TH2F* hCMQ[2];
  for (int i=0; i<2; ++i) {
    CMQ[i] = dynamic_cast<MonitorElementT<TNamed>*>(meCMCol_[i]);           
    hCMQ[i] = dynamic_cast<TH2F*> (CMQ[i]->operator->());  
  }
  
  gStyle->SetOptStat("");
  cCMQ->Divide(2,1);
  for (int i=0; i<2; ++i) {
    cCMQ->cd(i+1);
    gPad->SetGridx();
    gPad->SetGridy();
    hCMQ[i]->GetXaxis()->SetNdivisions(-104);
    hCMQ[i]->GetYaxis()->SetNdivisions(-104);
    hCMQ[i]->SetMinimum(-0.00000001);
    hCMQ[i]->SetMaximum(7.0);
    char tit[128]; sprintf(tit,"Plane %d",i+1);
    hCMQ[i]->SetTitle(tit);
    hCMQ[i]->Draw("col");
    gPad->Update();
    TPaveText *t = (TPaveText*) gPad->GetPrimitive("title");
    t->SetTextColor(4);
    t->SetTextSize(.1);
    t->SetBorderSize(0);
    t->SetX1NDC(0.00); t->SetX2NDC(1);
    t->SetY1NDC(0.93); t->SetY2NDC(1);
  }
  histName = htmlDir+"/PedestalCM_Quality.png";
  cCMQ->SaveAs(histName.c_str());  

  // Plot Mean and RMS
  MonitorElementT<TNamed>* Mean = dynamic_cast<MonitorElementT<TNamed>*>(meMean_);
  TH1F *hMean = dynamic_cast<TH1F*> (Mean->operator->());
  MonitorElementT<TNamed>* RMS  = dynamic_cast<MonitorElementT<TNamed>*>(meRMS_);
  TH1F *hRMS = dynamic_cast<TH1F*> (RMS->operator->());

  gStyle->SetOptStat(111110);
  cCM->Divide(2,1);
  cCM->cd(1);
  hMean->Draw();
  cCM->cd(2);
  hRMS->Draw();
  histName = htmlDir+"/PedestalCM_Mean_RMS.png";
  cCM->SaveAs(histName.c_str());

  // Show plots
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"PedestalCM_Mean_RMS.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"PedestalCM_Quality.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}
