#include "DQM/EcalPreshowerMonitorClient/interface/ESTDCCTClient.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElementBaseT.h"

#include "TStyle.h"
#include "TPaveText.h"

ESTDCCTClient::ESTDCCTClient(const ParameterSet& ps) {
  
  writeHisto_ = ps.getUntrackedParameter<bool>("writeHisto", true);
  writeHTML_  = ps.getUntrackedParameter<bool>("writeHTML", true);
  dumpRate_   = ps.getUntrackedParameter<int>("dumpRate", 100);
  outputFileName_ = ps.getUntrackedParameter<string>("outputFileName", "ESTDCCT");
  rootFolder_ = ps.getUntrackedParameter<string>("rootFolder", "");
  htmlDir_    = ps.getUntrackedParameter<string>("htmlDir","/preshower/DQM/TB");
  htmlName_   = ps.getUntrackedParameter<string>("htmlName","ESTDCCT.html");  
  sta_        = ps.getUntrackedParameter<bool>("RunStandalone", false);

  count_ = 0;
  run_ = -1;
  init_ = false;
  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

}

ESTDCCTClient::~ESTDCCTClient(){
}

void ESTDCCTClient::endJob(){
  
  Char_t runNum_s[50];
  sprintf(runNum_s, "%08d", run_);
  outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
  
  if (writeHTML_) {
    doQT();
    htmlOutput(run_, htmlDir_, htmlName_);
  }

  if (writeHisto_) dbe_->save(outputFile_);
  dbe_->rmdir("ES/QT/TDCCT");  

  if ( init_ ) this->cleanup();
}

void ESTDCCTClient::setup() {

   init_ = true;

}

void ESTDCCTClient::beginJob(const EventSetup& context){

  if (dbe_) {
    dbe_->setVerbose(1);
    dbe_->setCurrentFolder("ES/QT/TDCCT");
    dbe_->rmdir("ES/QT/TDCCT");
  }

}

void ESTDCCTClient::cleanup() {

  if (sta_) return;

  if (dbe_) {
    dbe_->setCurrentFolder("ES/QT/TDCCT");
  }

  init_ = false;

}

void ESTDCCTClient::analyze(const Event& e, const EventSetup& context){
	
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

void ESTDCCTClient::doQT() {

  MonitorElementT<TNamed>* meT;

  MonitorElement * meTDC = dbe_->get(getMEName("ES TDC"));
    
  if (meTDC) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meTDC);           
    hTDC_ = dynamic_cast<TH1F*> (meT->operator->());      
  }

  for (int i=0; i<2; ++i) {
    int zside = (i==0)?1:-1;

    for (int j=0; j<6; ++j) {

      MonitorElement *meTDCADCT = dbe_->get(getMEName(zside, j+1, 0, 1));
      if (meTDCADCT) {
	meT = dynamic_cast<MonitorElementT<TNamed>*>(meTDCADCT);
	hTDCADCT_[i][j] = dynamic_cast<TH2F*> (meT->operator->());
      }
      
      for (int k=0; k<3; ++k) {
	MonitorElement *meTDCADC = dbe_->get(getMEName(zside, j+1, k+1, 0));
	if (meTDCADC) {
	  meT = dynamic_cast<MonitorElementT<TNamed>*>(meTDCADC);
	  hTDCADC_[i][j][k] = dynamic_cast<TH2F*> (meT->operator->());
	}
      }
    }
  }
  
}

string ESTDCCTClient::getMEName(const string & meName) {
  
  string histoname = rootFolder_+"ES/ESTDCCTTask/"+meName; 
  
  return histoname;
  
}

string ESTDCCTClient::getMEName(const int & zside, const int & plane, const int & slot, const int & type) {
  
  Char_t hist[500];
  if (type == 0)
    sprintf(hist,"%sES/ESTDCCTTask/ES TDC ADC Z %d P %d Slot %d",rootFolder_.c_str(),zside,plane,slot);
  else if (type == 1)
    sprintf(hist,"%sES/ESTDCCTTask/ES TDC ADC Z %d P %d",rootFolder_.c_str(),zside,plane);
  
  return hist;
}

void ESTDCCTClient::htmlOutput(int run, string htmlDir, string htmlName) {
  
  cout<<"Going to output ESTDCCTClient html ..."<<endl;
  
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
  htmlFile << "  <title>Preshower DQM : TDCCTTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run Number / Num of Analyzed events :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;"<< count_ <<"</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task :&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">TDC</span></h2> " << endl;
  htmlFile << "<hr>" << endl;

  // Plot TDC
  string histName;
  gROOT->SetStyle("Plain");
  gStyle->SetStatW(0.3);
  gStyle->SetStatH(0.3);
  gStyle->SetOptStat("");
  gStyle->SetPalette(1, 0);
  gStyle->SetGridStyle(1);

  TCanvas *cTDC = new TCanvas("cTDC", "cTDC", 300, 300);
  cTDC->cd();
  char tit[128]; sprintf(tit,"TDC");
  hTDC_->SetTitle(tit);
  hTDC_->SetLineColor(6);
  hTDC_->Draw();
  gPad->Update();
  TPaveText *t = (TPaveText*) gPad->GetPrimitive("title");
  t->SetTextColor(4);
  t->SetTextSize(.1);
  t->SetBorderSize(0);
  t->SetX1NDC(0.00); t->SetX2NDC(1);
  t->SetY1NDC(0.93); t->SetY2NDC(1);
  
  histName = htmlDir+"/TDC.png";
  cTDC->SaveAs(histName.c_str());  

  htmlFile << "<img src=\"TDC.png\"></img>" << endl;

  TCanvas *cTDCADC1 = new TCanvas("cTDCADC1", "cTDCADC1", 1200, 1800);

  cTDCADC1->Divide(4,6);
  for (int i=0; i<6; ++i) {
    for (int j=0; j<3; j++) {
      cTDCADC1->cd((i*4)+j+1);
      sprintf(tit,"Box 1 P %d Slot %d", i+1, j+1);
      hTDCADC_[0][i][j]->SetTitle(tit);
      hTDCADC_[0][i][j]->Draw("col");
      gPad->Update();
      t = (TPaveText*) gPad->GetPrimitive("title");
      t->SetTextColor(4);
      t->SetTextSize(.1);
      t->SetBorderSize(0);
      t->SetX1NDC(0.00); t->SetX2NDC(1);
      t->SetY1NDC(0.93); t->SetY2NDC(1);
    }
    cTDCADC1->cd((i*4)+4);
    gPad->SetGridx();
    sprintf(tit,"Box 1 Plane %d", i+1);
    hTDCADCT_[0][i]->SetTitle(tit);
    hTDCADCT_[0][i]->GetXaxis()->SetNdivisions(-103);
    hTDCADCT_[0][i]->GetXaxis()->SetLabelSize(0.06);
    hTDCADCT_[0][i]->Draw("col");
    gPad->Update();
    t = (TPaveText*) gPad->GetPrimitive("title");
    t->SetTextColor(4);
    t->SetTextSize(.1);
    t->SetBorderSize(0);
    t->SetX1NDC(0.00); t->SetX2NDC(1);
    t->SetY1NDC(0.93); t->SetY2NDC(1);
  }
  
  histName = htmlDir+"/TDCADC_Box1.png";
  cTDCADC1->SaveAs(histName.c_str());  

  htmlFile << "<img src=\"TDCADC_Box1.png\"></img>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

