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
#include <TGraphErrors.h>

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

      Char_t tit[200];

      sprintf(tit, "%sES/ESOccupancyCTTask/Box1 Plane vs Strip, Current event",rootFolder_.c_str());
      MonitorElement *hitStrips1B= dbe_->get(tit);
      if (hitStrips1B) {
	meT = dynamic_cast<MonitorElementT<TNamed>*>(hitStrips1B);
	hStrips1B_ = dynamic_cast<TH2F*> (meT->operator->());
      }

      sprintf(tit, "%sES/ESOccupancyCTTask/Box1 Plane vs Sensor, Current event",rootFolder_.c_str());
      MonitorElement *hitSensors1B= dbe_->get(tit);
      if (hitSensors1B) {
	meT = dynamic_cast<MonitorElementT<TNamed>*>(hitSensors1B);
	hSensors1B_ = dynamic_cast<TH2F*> (meT->operator->());
      }

      sprintf(tit, "%sES/ESOccupancyCTTask/Box2 Plane vs Strip, Current event",rootFolder_.c_str());
      MonitorElement *hitStrips2B= dbe_->get(tit);
      if (hitStrips2B) {
	meT = dynamic_cast<MonitorElementT<TNamed>*>(hitStrips2B);
	hStrips2B_ = dynamic_cast<TH2F*> (meT->operator->());
      }

      sprintf(tit, "%sES/ESOccupancyCTTask/Box2 Plane vs Sensor, Current event",rootFolder_.c_str());
      MonitorElement *hitSensors2B= dbe_->get(tit);
      if (hitSensors2B) {
	meT = dynamic_cast<MonitorElementT<TNamed>*>(hitSensors2B);
	hSensors2B_ = dynamic_cast<TH2F*> (meT->operator->());
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

  delete cE; cE=0;

  //Read info for current event 
  int hit1_strips[6][64], hit1_sensors[6][5]; //Z=1 (1st box)
  int hit2_strips[6][64], hit2_sensors[6][5]; //Z=1 (2nd box)


  for(int i=0;i<6;i++){
     for(int j=0;j<64;j++){
       hit1_strips[i][j]=(int)hStrips1B_->GetCellContent(j+1,i+1);
       hit2_strips[i][j]=(int)hStrips2B_->GetCellContent(j+1,i+1);
     }
     for(int j=0;j<5;j++){
       hit1_sensors[i][j]=(int)hSensors1B_->GetCellContent(j+1,i+1);
       hit2_sensors[i][j]=(int)hSensors2B_->GetCellContent(j+1,i+1);
     }
  }


  float strip_no[64], lad1[64], lad2[64], lad3[64], lad4[64], lad5[64], lad6[64];
  float sensor_no[5],slad1[5], slad2[5], slad3[5], slad4[5], slad5[5], slad6[5];
  
  float hits1[64], hits2[64], hits3[64], hits4[64], hits5[64], hits6[64];
  float shits1[5], shits2[5], shits3[5], shits4[5], shits5[5], shits6[5];

  float hits1_2[64], hits2_2[64], hits3_2[64], hits4_2[64], hits5_2[64], hits6_2[64];
  float shits1_2[5], shits2_2[5], shits3_2[5], shits4_2[5], shits5_2[5], shits6_2[5];


  for(int i=0;i<64;++i){
    strip_no[i]=i;
    lad1[i]=6; lad2[i]=5; lad3[i]=4; lad4[i]=3; lad5[i]=2; lad6[i]=1;
    hits1[i]=-10; hits2[i]=-10; hits3[i]=-10; hits4[i]=-10; hits5[i]=-10; hits6[i]=-10;
    hits1_2[i]=-10; hits2_2[i]=-10; hits3_2[i]=-10; hits4_2[i]=-10; hits5_2[i]=-10; hits6_2[i]=-10;
  }

  for(int i=0;i<5;++i){
    sensor_no[i]=i+0.5;
    slad1[i]=6; slad2[i]=5; slad3[i]=4; slad4[i]=3; slad5[i]=2; slad6[i]=1;
    shits1[i]=-10; shits2[i]=-10; shits3[i]=-10; shits4[i]=-10; shits5[i]=-10; shits6[i]=-10;
    shits1_2[i]=-10; shits2_2[i]=-10; shits3_2[i]=-10; shits4_2[i]=-10; shits5_2[i]=-10; shits6_2[i]=-10;
  }

  TGraph *l1 = new TGraph(64,strip_no,lad1);
  TGraph *l2 = new TGraph(64,strip_no,lad2);
  TGraph *l3 = new TGraph(64,strip_no,lad3);
  TGraph *l4 = new TGraph(64,strip_no,lad4);
  TGraph *l5 = new TGraph(64,strip_no,lad5);
  TGraph *l6 = new TGraph(64,strip_no,lad6);

  TGraph *sl1 = new TGraph(5,sensor_no,slad1);
  TGraph *sl2 = new TGraph(5,sensor_no,slad2);
  TGraph *sl3 = new TGraph(5,sensor_no,slad3);
  TGraph *sl4 = new TGraph(5,sensor_no,slad4);
  TGraph *sl5 = new TGraph(5,sensor_no,slad5);
  TGraph *sl6 = new TGraph(5,sensor_no,slad6);



   //First Box setup
  for(int j=0;j<64;j++){
    if(hit1_strips[0][j]>0) hits1[j]=1; 
    if(hit1_strips[1][j]>0) hits2[j]=2;
    if(hit1_strips[2][j]>0) hits3[j]=3;
    if(hit1_strips[3][j]>0) hits4[j]=4;
    if(hit1_strips[4][j]>0) hits5[j]=5;
    if(hit1_strips[5][j]>0) hits6[j]=6;
  }
  for(int j=0;j<5;j++){
    if(hit1_sensors[0][j]>0) shits1[j]=1;
    if(hit1_sensors[1][j]>0) shits2[j]=2;
    if(hit1_sensors[2][j]>0) shits3[j]=3;
    if(hit1_sensors[3][j]>0) shits4[j]=4;
    if(hit1_sensors[4][j]>0) shits5[j]=5;
    if(hit1_sensors[5][j]>0) shits6[j]=6;
   }

 
  TGraph *hitL1 = new TGraph(64,strip_no,hits1);
  TGraph *hitL2 = new TGraph(64,strip_no,hits2);
  TGraph *hitL3 = new TGraph(64,strip_no,hits3);
  TGraph *hitL4 = new TGraph(64,strip_no,hits4);
  TGraph *hitL5 = new TGraph(64,strip_no,hits5);
  TGraph *hitL6 = new TGraph(64,strip_no,hits6);

  TGraph *shitL1 = new TGraph(5,sensor_no,shits1);
  TGraph *shitL2 = new TGraph(5,sensor_no,shits2);
  TGraph *shitL3 = new TGraph(5,sensor_no,shits3);
  TGraph *shitL4 = new TGraph(5,sensor_no,shits4);
  TGraph *shitL5 = new TGraph(5,sensor_no,shits5);
  TGraph *shitL6 = new TGraph(5,sensor_no,shits6);


  l1->SetMarkerStyle(21);
  l1->SetMarkerSize(0.2);
  l2->SetMarkerStyle(21);
  l2->SetMarkerSize(0.2);
  l3->SetMarkerStyle(21);
  l3->SetMarkerSize(0.2);
  l4->SetMarkerStyle(21);
  l4->SetMarkerSize(0.2);
  l5->SetMarkerStyle(21);
  l5->SetMarkerSize(0.2);
  l6->SetMarkerStyle(21);
  l6->SetMarkerSize(0.2);

  l1->SetMaximum(7);
  l1->SetMinimum(0);
  l1->SetTitle("Tracking Strips Box1");
  l1->GetXaxis()->SetTitle("Strip Number");
  l1->GetYaxis()->SetTitle("Ladder Number");

  sl1->SetMarkerStyle(21);
  sl1->SetMarkerSize(1);
  sl2->SetMarkerStyle(21);
  sl2->SetMarkerSize(1);
  sl3->SetMarkerStyle(21);
  sl3->SetMarkerSize(1);
  sl4->SetMarkerStyle(21);
  sl4->SetMarkerSize(1);
  sl5->SetMarkerStyle(21);
  sl5->SetMarkerSize(1);
  sl6->SetMarkerStyle(21);
  sl6->SetMarkerSize(1);

  sl1->SetMaximum(7);
  sl1->SetMinimum(0);
  sl1->SetTitle("Tracking Sensors Box1");
  sl1->GetXaxis()->SetTitle("Sensor Number");
  sl1->GetYaxis()->SetTitle("Ladder Number");


  hitL1->SetMarkerStyle(21);
  hitL1->SetMarkerColor(2);
  hitL1->SetMarkerSize(0.5);
  hitL2->SetMarkerStyle(21);
  hitL2->SetMarkerColor(2);
  hitL2->SetMarkerSize(0.5);
  hitL3->SetMarkerStyle(21);
  hitL3->SetMarkerColor(2);
  hitL3->SetMarkerSize(0.5);
  hitL4->SetMarkerStyle(21);
  hitL4->SetMarkerColor(2);
  hitL4->SetMarkerSize(0.5);
  hitL5->SetMarkerStyle(21);
  hitL5->SetMarkerColor(2);
  hitL5->SetMarkerSize(0.5);
  hitL6->SetMarkerStyle(21);
  hitL6->SetMarkerColor(2);
  hitL6->SetMarkerSize(0.5);

  shitL1->SetMarkerStyle(21);
  shitL1->SetMarkerColor(2);
  shitL1->SetMarkerSize(2);
  shitL2->SetMarkerStyle(21);
  shitL2->SetMarkerColor(2);
  shitL2->SetMarkerSize(2);
  shitL3->SetMarkerStyle(21);
  shitL3->SetMarkerColor(2);
  shitL3->SetMarkerSize(2);
  shitL4->SetMarkerStyle(21);
  shitL4->SetMarkerColor(2);
  shitL4->SetMarkerSize(2);
  shitL5->SetMarkerStyle(21);
  shitL5->SetMarkerColor(2);
  shitL5->SetMarkerSize(2);
  shitL6->SetMarkerStyle(21);
  shitL6->SetMarkerColor(2);
  shitL6->SetMarkerSize(2);


  TCanvas *cv = new TCanvas("cv", "cv", 900, 450);
  cv->Divide(2,1); 
  //cv->SetFillColor(42);

  cv->cd(1);
  l1->Draw("AP");
  l2->Draw("P");
  l3->Draw("P");
  l4->Draw("P");
  l5->Draw("P");
  l6->Draw("P");
  hitL1->Draw("P");
  hitL2->Draw("P");
  hitL3->Draw("P");
  hitL4->Draw("P");
  hitL5->Draw("P");
  hitL6->Draw("P");

  cv->cd(2);
  sl1->Draw("AP");
  sl2->Draw("P");
  sl3->Draw("P");
  sl4->Draw("P");
  sl5->Draw("P");
  sl6->Draw("P");
  shitL1->Draw("P");
  shitL2->Draw("P");
  shitL3->Draw("P");
  shitL4->Draw("P");
  shitL5->Draw("P");
  shitL6->Draw("P");
  histName = htmlDir+"/Box1CurEvent.png";
  cv->SaveAs(histName.c_str());  




   /// Second Box setup   
  for(int j=0;j<64;j++){
    if(hit2_strips[0][j]>0) hits1_2[j]=1; 
    if(hit2_strips[1][j]>0) hits2_2[j]=2;
    if(hit2_strips[2][j]>0) hits3_2[j]=3;
    if(hit2_strips[3][j]>0) hits4_2[j]=4;
    if(hit2_strips[4][j]>0) hits5_2[j]=5;
    if(hit2_strips[5][j]>0) hits6_2[j]=6;
  }
  for(int j=0;j<5;j++){
    if(hit2_sensors[0][j]>0) shits1_2[j]=1;
    if(hit2_sensors[1][j]>0) shits2_2[j]=2;
    if(hit2_sensors[2][j]>0) shits3_2[j]=3;
    if(hit2_sensors[3][j]>0) shits4_2[j]=4;
    if(hit2_sensors[4][j]>0) shits5_2[j]=5;
    if(hit2_sensors[5][j]>0) shits6_2[j]=6;
   }


  TGraph *hitL1_2 = new TGraph(64,strip_no,hits1_2);
  TGraph *hitL2_2 = new TGraph(64,strip_no,hits2_2);
  TGraph *hitL3_2 = new TGraph(64,strip_no,hits3_2);
  TGraph *hitL4_2 = new TGraph(64,strip_no,hits4_2);
  TGraph *hitL5_2 = new TGraph(64,strip_no,hits5_2);
  TGraph *hitL6_2 = new TGraph(64,strip_no,hits6_2);

  TGraph *shitL1_2 = new TGraph(5,sensor_no,shits1_2);
  TGraph *shitL2_2 = new TGraph(5,sensor_no,shits2_2);
  TGraph *shitL3_2 = new TGraph(5,sensor_no,shits3_2);
  TGraph *shitL4_2 = new TGraph(5,sensor_no,shits4_2);
  TGraph *shitL5_2 = new TGraph(5,sensor_no,shits5_2);
  TGraph *shitL6_2 = new TGraph(5,sensor_no,shits6_2);


  l1->SetTitle("Tracking Strips Box2");
  l1->GetXaxis()->SetTitle("Strip Number");
  l1->GetYaxis()->SetTitle("Ladder Number");

  sl1->SetTitle("Tracking Sensors Box2");
  sl1->GetXaxis()->SetTitle("Sensor Number");
  sl1->GetYaxis()->SetTitle("Ladder Number");

  hitL1_2->SetMarkerStyle(21);
  hitL1_2->SetMarkerColor(2);
  hitL1_2->SetMarkerSize(0.5);
  hitL2_2->SetMarkerStyle(21);
  hitL2_2->SetMarkerColor(2);
  hitL2_2->SetMarkerSize(0.5);
  hitL3_2->SetMarkerStyle(21);
  hitL3_2->SetMarkerColor(2);
  hitL3_2->SetMarkerSize(0.5);
  hitL4_2->SetMarkerStyle(21);
  hitL4_2->SetMarkerColor(2);
  hitL4_2->SetMarkerSize(0.5);
  hitL5_2->SetMarkerStyle(21);
  hitL5_2->SetMarkerColor(2);
  hitL5_2->SetMarkerSize(0.5);
  hitL6_2->SetMarkerStyle(21);
  hitL6_2->SetMarkerColor(2);
  hitL6_2->SetMarkerSize(0.5);

  shitL1_2->SetMarkerStyle(21);
  shitL1_2->SetMarkerColor(2);
  shitL1_2->SetMarkerSize(2);
  shitL2_2->SetMarkerStyle(21);
  shitL2_2->SetMarkerColor(2);
  shitL2_2->SetMarkerSize(2);
  shitL3_2->SetMarkerStyle(21);
  shitL3_2->SetMarkerColor(2);
  shitL3_2->SetMarkerSize(2);
  shitL4_2->SetMarkerStyle(21);
  shitL4_2->SetMarkerColor(2);
  shitL4_2->SetMarkerSize(2);
  shitL5_2->SetMarkerStyle(21);
  shitL5_2->SetMarkerColor(2);
  shitL5_2->SetMarkerSize(2);
  shitL6_2->SetMarkerStyle(21);
  shitL6_2->SetMarkerColor(2);
  shitL6_2->SetMarkerSize(2);

  TCanvas *cv2 = new TCanvas("cv2", "cv2", 900, 450);
  cv2->Divide(2,1); 
  cv2->cd(1);
  l1->Draw("AP");
  l2->Draw("P");
  l3->Draw("P");
  l4->Draw("P");
  l5->Draw("P");
  l6->Draw("P");
  hitL1_2->Draw("P");
  hitL2_2->Draw("P");
  hitL3_2->Draw("P");
  hitL4_2->Draw("P");
  hitL5_2->Draw("P");
  hitL6_2->Draw("P");

  cv2->cd(2);
  sl1->Draw("AP");
  sl2->Draw("P");
  sl3->Draw("P");
  sl4->Draw("P");
  sl5->Draw("P");
  sl6->Draw("P");
  shitL1_2->Draw("P");
  shitL2_2->Draw("P");
  shitL3_2->Draw("P");
  shitL4_2->Draw("P");
  shitL5_2->Draw("P");
  shitL6_2->Draw("P");
  histName = htmlDir+"/Box2CurEvent.png";
  cv2->SaveAs(histName.c_str());  

  delete cv; cv=0;
  delete cv2; cv2=0;
  delete l1; l1=0;
  delete l2; l2=0;
  delete l3; l3=0;
  delete l4; l4=0;
  delete l5; l5=0;
  delete l6; l6=0;
  delete sl1; sl1=0;
  delete sl2; sl2=0;
  delete sl3; sl3=0;
  delete sl4; sl4=0;
  delete sl5; sl5=0;
  delete sl6; sl6=0;
  delete hitL1; hitL1=0;
  delete hitL2; hitL2=0;
  delete hitL3; hitL3=0;
  delete hitL4; hitL4=0;
  delete hitL5; hitL5=0;
  delete hitL6; hitL6=0;
  delete shitL1; shitL1=0;
  delete shitL2; shitL2=0;
  delete shitL3; shitL3=0;
  delete shitL4; shitL4=0;
  delete shitL5; shitL5=0;
  delete shitL6; shitL6=0;
  delete hitL1_2; hitL1_2=0;
  delete hitL2_2; hitL2_2=0;
  delete hitL3_2; hitL3_2=0;
  delete hitL4_2; hitL4_2=0;
  delete hitL5_2; hitL5_2=0;
  delete hitL6_2; hitL6_2=0;
  delete shitL1_2; shitL1_2=0;
  delete shitL2_2; shitL2_2=0;
  delete shitL3_2; shitL3_2=0;
  delete shitL4_2; shitL4_2=0;
  delete shitL5_2; shitL5_2=0;
  delete shitL6_2; shitL6_2=0;

  htmlFile << "<img src=\"EnergySpectrum.png\"></img>" << endl;
  htmlFile << "<img src=\"Occupancy1D.png\"></img>" << endl;
  htmlFile << "<img src=\"Occupancy2D.png\"></img>" << endl;
  htmlFile << "<img src=\"Box1CurEvent.png\"></img>" << endl;
  htmlFile << "<img src=\"Box2CurEvent.png\"></img>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

