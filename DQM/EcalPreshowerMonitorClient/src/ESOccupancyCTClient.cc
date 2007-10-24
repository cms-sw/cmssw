#include "DQM/EcalPreshowerMonitorClient/interface/ESOccupancyCTClient.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElementBaseT.h"

#include "TStyle.h"
#include "TPaveText.h"

ESOccupancyCTClient::ESOccupancyCTClient(const ParameterSet& ps) {
  
  writeHisto_ = ps.getUntrackedParameter<bool>("writeHisto", true);
  writeHTML_  = ps.getUntrackedParameter<bool>("writeHTML", true);
  dumpRate_   = ps.getUntrackedParameter<int>("dumpRate", 100);
  outputFileName_ = ps.getUntrackedParameter<string>("outputFileName", "ESOccupancyCT");
  rootFolder_ = ps.getUntrackedParameter<string>("rootFolder", "");
  htmlDir_    = ps.getUntrackedParameter<string>("htmlDir","/preshower/DQM/TB");
  htmlName_   = ps.getUntrackedParameter<string>("htmlName","ESOccupancyCT.html");  
  sta_        = ps.getUntrackedParameter<bool>("RunStandalone", false);

  count_ = 0;
  run_ = -1;
  init_ = false;
  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

}

ESOccupancyCTClient::~ESOccupancyCTClient(){
}

void ESOccupancyCTClient::endJob(){
  
  Char_t runNum_s[50];
  sprintf(runNum_s, "%08d", run_);
  outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
  
  if (writeHTML_) {
    doQT();
    htmlOutput(run_, htmlDir_, htmlName_);
  }

  if (writeHisto_) dbe_->save(outputFile_);
  dbe_->rmdir("ES/QT/OccupancyCT");  

  if ( init_ ) this->cleanup();
}

void ESOccupancyCTClient::setup() {

   init_ = true;

}

void ESOccupancyCTClient::beginJob(const EventSetup& context){

  if (dbe_) {
    dbe_->setVerbose(1);
    dbe_->setCurrentFolder("ES/QT/OccupancyCT");
    dbe_->rmdir("ES/QT/OccupancyCT");
  }

}

void ESOccupancyCTClient::cleanup() {

  if (sta_) return;

  if (dbe_) {
    dbe_->setCurrentFolder("ES/QT/OccupancyCT");
  }

  init_ = false;

}

void ESOccupancyCTClient::analyze(const Event& e, const EventSetup& context){
	
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

void ESOccupancyCTClient::doQT() {

  MonitorElementT<TNamed>* meT;

  for (int i=0; i<2; ++i) {
    for (int j=0; j<6; ++j) {

      MonitorElement *meEnergy= dbe_->get(getMEName(i+1, j+1, 0));
      if (meEnergy) {
	meT = dynamic_cast<MonitorElementT<TNamed>*>(meEnergy);
	hEnergy_[i][j] = dynamic_cast<TH1F*> (meT->operator->());
      }

      MonitorElement *meOccupancy1D= dbe_->get(getMEName(i+1, j+1, 1));
      if (meOccupancy1D) {
	meT = dynamic_cast<MonitorElementT<TNamed>*>(meOccupancy1D);
	hOccupancy1D_[i][j] = dynamic_cast<TH1F*> (meT->operator->());
      }

      MonitorElement *meOccupancy2D= dbe_->get(getMEName(i+1, j+1, 2));
      if (meOccupancy2D) {
	meT = dynamic_cast<MonitorElementT<TNamed>*>(meOccupancy2D);
	hOccupancy2D_[i][j] = dynamic_cast<TH2F*> (meT->operator->());
      }
      
    }
  }
  
}

string ESOccupancyCTClient::getMEName(const int & zside, const int & plane, const int & type) {
  
  Char_t hist[500];
  if (type == 0)
    sprintf(hist,"%sES/ESOccupancyCTTask/ES Energy Box %d P %d",rootFolder_.c_str(),zside,plane);
  else if (type == 1)
    sprintf(hist,"%sES/ESOccupancyCTTask/ES Occupancy 1D Box %d P %d",rootFolder_.c_str(),zside,plane);
  else if (type == 2)
    sprintf(hist,"%sES/ESOccupancyCTTask/ES Occupancy 2D Box %d P %d",rootFolder_.c_str(),zside,plane);
  
  return hist;
}

void ESOccupancyCTClient::htmlOutput(int run, string htmlDir, string htmlName) {
  
  cout<<"Going to output ESOccupancyCTClient html ..."<<endl;
  
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
  htmlFile << "  <title>Preshower DQM : OccupancyCTTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run Number / Num of Analyzed events :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;"<< count_ <<"</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task :&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Energy Spectrum and Occupancy</span></h2> " << endl;
  htmlFile << "<hr>" << endl;

  // Plot Occupancy
  string histName;
  gROOT->SetStyle("Plain");
  gStyle->SetStatW(0.3);
  gStyle->SetStatH(0.3);
  gStyle->SetPalette(1, 0);
  gStyle->SetGridStyle(1);

  TCanvas *cE = new TCanvas("cE", "cE", 1500, 300);
  gStyle->SetOptStat(111110);
  cE->Divide(5,1);
  //cE->Divide(6,2);
  for (int i=0; i<1; ++i) {
    for (int j=0; j<5; ++j) {
      cE->cd(j+1+i*6);
      hEnergy_[i][j]->GetXaxis()->SetTitle("keV");
      hEnergy_[i][j]->Draw();
    }
  }
  histName = htmlDir+"/EnergySpectrum.png";
  cE->SaveAs(histName.c_str());  

  gStyle->SetOptStat(111110);
  for (int i=0; i<1; ++i) {
    for (int j=0; j<5; ++j) {
      cE->cd(j+1+i*6);
      hOccupancy1D_[i][j]->Draw();
    }
  }
  histName = htmlDir+"/Occupancy1D.png";
  cE->SaveAs(histName.c_str());  

  gStyle->SetOptStat("");
  for (int i=0; i<1; ++i) {
    for (int j=0; j<5; ++j) {
      cE->cd(j+1+i*6);
      gPad->SetGridx();
      gPad->SetGridy();
      hOccupancy2D_[i][j]->GetXaxis()->SetNdivisions(-104);
      hOccupancy2D_[i][j]->GetYaxis()->SetNdivisions(-105);
      hOccupancy2D_[i][j]->Draw("colz");
    }
  }
  histName = htmlDir+"/Occupancy2D.png";
  cE->SaveAs(histName.c_str());  

  htmlFile << "<img src=\"EnergySpectrum.png\"></img>" << endl;
  htmlFile << "<img src=\"Occupancy1D.png\"></img>" << endl;
  htmlFile << "<img src=\"Occupancy2D.png\"></img>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

