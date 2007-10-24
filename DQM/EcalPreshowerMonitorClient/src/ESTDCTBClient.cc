#include "DQM/EcalPreshowerMonitorClient/interface/ESTDCTBClient.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElementBaseT.h"

#include "TStyle.h"
#include "TPaveText.h"

ESTDCTBClient::ESTDCTBClient(const ParameterSet& ps) {
  
  writeHisto_ = ps.getUntrackedParameter<bool>("writeHisto", true);
  writeHTML_  = ps.getUntrackedParameter<bool>("writeHTML", true);
  dumpRate_   = ps.getUntrackedParameter<int>("dumpRate", 100);
  outputFileName_ = ps.getUntrackedParameter<string>("outputFileName", "ESTDCTB");
  rootFolder_ = ps.getUntrackedParameter<string>("rootFolder", "");
  htmlDir_    = ps.getUntrackedParameter<string>("htmlDir","/preshower/DQM/TB");
  htmlName_   = ps.getUntrackedParameter<string>("htmlName","ESTDCTB.html");  
  sta_        = ps.getUntrackedParameter<bool>("RunStandalone", false);

  count_ = 0;
  run_ = -1;
  init_ = false;
  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

}

ESTDCTBClient::~ESTDCTBClient(){
}

void ESTDCTBClient::endJob(){
  
  Char_t runNum_s[50];
  sprintf(runNum_s, "%08d", run_);
  outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
  
  if (writeHTML_) {
    doQT();
    htmlOutput(run_, htmlDir_, htmlName_);
  }

  if (writeHisto_) dbe_->save(outputFile_);
  dbe_->rmdir("ES/QT/TDCTB");  

  if ( init_ ) this->cleanup();
}

void ESTDCTBClient::setup() {

   init_ = true;

}

void ESTDCTBClient::beginJob(const EventSetup& context){

  if (dbe_) {
    dbe_->setVerbose(1);
    dbe_->setCurrentFolder("ES/QT/TDCTB");
    dbe_->rmdir("ES/QT/TDCTB");
  }

}

void ESTDCTBClient::cleanup() {

  if (sta_) return;

  if (dbe_) {
    dbe_->setCurrentFolder("ES/QT/TDCTB");
  }

  init_ = false;

}

void ESTDCTBClient::analyze(const Event& e, const EventSetup& context){
	
  if (! init_) this->setup();

  int runNum = e.id().run();
  Char_t runNum_s[50];
      
  if (runNum != run_) { 
    
    if (run_ > 0) {

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

    sprintf(runNum_s, "%08d", run_);
    outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
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

void ESTDCTBClient::doQT() {

  MonitorElementT<TNamed>* meT;

  for (int i=0; i<2; ++i) {
    for (int j=0; j<3; ++j) {
      
      MonitorElement *meADC = dbe_->get(getMEName(i+1, j+1));
      if (meADC) {
	meT = dynamic_cast<MonitorElementT<TNamed>*>(meADC);
	hADC_[i][j] = dynamic_cast<TH1F*> (meT->operator->());
      }
    }
  }
  
}

string ESTDCTBClient::getMEName(const string & meName) {
  
  string histoname = rootFolder_+"ES/ESTDCTBTask/"+meName; 
  
  return histoname;
  
}

string ESTDCTBClient::getMEName(const int & plane, const int & slot) {
  
  Char_t hist[500];
  sprintf(hist,"%sES/ESTDCTBTask/ES ADC Z 1 P %d Slot %d",rootFolder_.c_str(),plane,slot);
  
  return hist;
}

void ESTDCTBClient::htmlOutput(int run, string htmlDir, string htmlName) {
  
  cout<<"Going to output ESTDCTBClient html ..."<<endl;
  
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
  htmlFile << "  <title>Preshower DQM : TDCTBTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run Number / Num of Analyzed events :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;"<< count_ <<"</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task :&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">TDC</span></h2> " << endl;
  htmlFile << "<hr>" << endl;

  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1, 0);
  gStyle->SetStatW(0.3);
  gStyle->SetStatH(0.3);
  gStyle->SetGridStyle(1);

  // Plot TDC
  string histName;
  Char_t tit[200];

  TCanvas *cADC = new TCanvas("cADC", "cADC", 1200, 800);
  gStyle->SetOptStat(111110);
  cADC->Divide(3,2);
  for (int i=0; i<2; ++i) {
    for (int j=0; j<3; ++j) {
      cADC->cd(1+(i*3)+j);
      gPad->SetLogy(1);
      sprintf(tit,"P %d Slot %d", i+1, j+1);
      hADC_[i][j]->SetTitle(tit);
      hADC_[i][j]->Draw();
      gPad->Update();
      TPaveText *t = (TPaveText*) gPad->GetPrimitive("title");
      t->SetTextColor(4);
      t->SetTextSize(.1);
      t->SetBorderSize(0);
      t->SetX1NDC(0.00); t->SetX2NDC(1);
      t->SetY1NDC(0.93); t->SetY2NDC(1);
    }
  }
  
  histName = htmlDir+"/ESADC.png";
  cADC->SaveAs(histName.c_str());  

  htmlFile << "<img src=\"ESADC.png\"></img>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

