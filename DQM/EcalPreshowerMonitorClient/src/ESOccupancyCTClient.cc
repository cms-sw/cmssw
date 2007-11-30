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

#include "TApplication.h"
#include "TGeoTrack.h"
#include "TMarker3DBox.h"
#include "TAxis3D.h"
#include "TGeoManager.h"
#include "TView.h"

ESOccupancyCTClient::ESOccupancyCTClient(const ParameterSet& ps) {

  writeHisto_ = ps.getUntrackedParameter<bool>("writeHisto", true);
  writeHTML_  = ps.getUntrackedParameter<bool>("writeHTML", true);
  dumpRate_   = ps.getUntrackedParameter<int>("dumpRate", 100);
  outputFileName_ = ps.getUntrackedParameter<string>("outputFileName", "ESOccupancyCT");
  rootFolder_ = ps.getUntrackedParameter<string>("rootFolder", "");
  htmlDir_    = ps.getUntrackedParameter<string>("htmlDir","/preshower/DQM/TB");
  htmlName_   = ps.getUntrackedParameter<string>("htmlName","ESOccupancyCT.html");  
  sta_        = ps.getUntrackedParameter<bool>("RunStandalone", false);
  x11_        = ps.getUntrackedParameter<bool>("InitializeX11", false);

  count_ = 0;
  run_ = -1;
  init_ = false;

  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  if (x11_) {
    // If this block is executed, the x11 interface is initialized
    // and the created canvases are displayed at run time.
    int argc=0;
    char *argv[1];
    theApp_= new TApplication("App", &argc, argv);
  }
  else
    theApp_= 0;

  CRtd(); // setup the track display

  // A TList that contains the current track objects
  currentTrackList_ = new TList();

}

ESOccupancyCTClient::~ESOccupancyCTClient(){
  if (theApp_) delete theApp_;
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

  Char_t tit[512];

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
  htmlFile << "<img src=\"Box1TrackDisplay.png\"></img>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  // Plot the track display canvas and save it to png
  PlotTrackDisplay();
  histName = htmlDir+"/Box1TrackDisplay.png";
  CRtd_canvas_->cd(0);
  CRtd_canvas_->SaveAs(histName.c_str());  


  htmlFile.close();

}

//==========================================================================
//==========================================================================
//==========================================================================


void ESOccupancyCTClient::PlotTrackDisplay()
{
  // Get the monitored data...
  char tit[512];
  string tmp;

  sprintf(tit, "%sES/ESOccupancyCTTask/%s",rootFolder_.c_str(),"meTrack_Npoints");
  tmp=dbe_->get(tit)->valueString().substr(2);
  int Npoints=atoi(tmp.c_str());
  //cout << "Npoints=" << Npoints << "   tmp=" << tmp << endl;

  int Nhits_lad[6];       // Nhits_lad[i]  : gives the number of hits in ladder i
  for (int i=0; i<6; i++) {
    sprintf(tit, "%sES/ESOccupancyCTTask/me_Nhits_lad%d",rootFolder_.c_str(),i);
    tmp=dbe_->get(tit)->valueString().substr(2);
    Nhits_lad[i]=atoi(tmp.c_str());
    //cout << "Nhits_lad[" << i << "]=" << Nhits_lad[i] << "   tmp=" << tmp << endl;
  }

  MonitorElementT<TNamed>* meT;
  MonitorElement *me;
  TH2F *me_hit_x=0;
  TH2F *me_hit_y=0;
  sprintf(tit, "%sES/ESOccupancyCTTask/me_hit_x",rootFolder_.c_str());
  me=dbe_->get(tit);
  if (me) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(me);
    me_hit_x = dynamic_cast<TH2F*> (meT->operator->());
  }
  sprintf(tit, "%sES/ESOccupancyCTTask/me_hit_y",rootFolder_.c_str());
  me=dbe_->get(tit);
  if (me) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(me);
    me_hit_y = dynamic_cast<TH2F*> (meT->operator->());
  }
  //cout << "me_hit_x=" << me_hit_x << "    me_hit_y=" << me_hit_y << endl;

  float hit_x[6][200];
  float hit_y[6][200];
  for (int i=0; i<6; i++) {
    for (int j=0; j<Nhits_lad[i]; j++) {
      hit_x[i][j]=me_hit_x->GetBinContent(i,j);
      hit_y[i][j]=me_hit_y->GetBinContent(i,j);
    }
  }

  int best_hit[6];
  float best_Px[6], best_Py[6], best_Pz[6], best_par[6];
  for (int i=0; i<6; i++) {
    sprintf(tit, "%sES/ESOccupancyCTTask/meTrack_hit%d",rootFolder_.c_str(),i);
    tmp=dbe_->get(tit)->valueString().substr(2);
    best_hit[i]=atoi(tmp.c_str());

    //cout << "best_hit[" << i << "]=" << best_hit[i] << "   tmp=" << tmp << endl;
    sprintf(tit, "%sES/ESOccupancyCTTask/meTrack_Px%d",rootFolder_.c_str(),i);
    tmp=dbe_->get(tit)->valueString().substr(2);
    best_Px[i]=atof(tmp.c_str());

    //cout << "best_Px[" << i << "]=" << best_Px[i] << "   tmp=" << tmp << endl;
    sprintf(tit, "%sES/ESOccupancyCTTask/meTrack_Py%d",rootFolder_.c_str(),i);
    tmp=dbe_->get(tit)->valueString().substr(2);
    best_Py[i]=atof(tmp.c_str());
    //cout << "best_Py[" << i << "]=" << best_Py[i] << "   tmp=" << tmp << endl;

    sprintf(tit, "%sES/ESOccupancyCTTask/meTrack_Pz%d",rootFolder_.c_str(),i);
    tmp=dbe_->get(tit)->valueString().substr(2);
    best_Pz[i]=atof(tmp.c_str());
    //cout << "best_Pz[" << i << "]=" << best_Pz[i] << "   tmp=" << tmp << endl;

    sprintf(tit, "%sES/ESOccupancyCTTask/meTrack_par%d",rootFolder_.c_str(),i);
    tmp=dbe_->get(tit)->valueString().substr(2);
    best_par[i]=atof(tmp.c_str());
    //cout << "best_par[" << i << "]=" << best_par[i] << "   tmp=" << tmp << endl;
  }

  // delete the previous track and hits
  currentTrackList_->Delete();

  Double_t DZ=39.0; // all sensors Z projection length (cm)

  // place off-track hits in blue...
  TMarker3DBox *mm_off[6][200];
  for (int i=0; i<6; i++) {
    for (int j=0; j<Nhits_lad[i]; j++) {
      if (j!=best_hit[i]) {
	mm_off[i][j] = new TMarker3DBox(hit_x[i][j], hit_y[i][j], i*DZ/5-DZ/2, 0.1,0.1,0.1, 45,45);
	currentTrackList_->Add(mm_off[i][j]);
	mm_off[i][j]->SetLineColor(4);
	mm_off[i][j]->SetLineWidth(8);
      }
      else {
	mm_off[i][j]=0;
      }
    }
  }

  // Check if no track was found
  int sum=0, TrackFound=1;
  for (int i=0; i<6; i++) sum+=best_hit[i];
  if (sum==-6) TrackFound=0;

  TMarker3DBox *mm[6];
  TGeoTrack *track=0;
  if (TrackFound) {
    // place track hits in red...
    for (int i=0; i<Npoints; i++) {
      mm[i] = new TMarker3DBox(best_Px[i], best_Py[i], best_Pz[i], 0.1,0.1,0.1, 45,45);
      currentTrackList_->Add(mm[i]);
      mm[i]->SetLineColor(2);
      mm[i]->SetLineWidth(8);
    }

    // make track
    Double_t x, y, z, t;
    track = new TGeoTrack();
    currentTrackList_->Add(track);
    track->SetLineColor(2);
    track->SetLineWidth(1);
    z=-50.0;  // 1st track point
    t=(z-best_par[2])/best_par[5];
    x=best_par[0]+t*best_par[3];
    y=best_par[1]+t*best_par[4];
    track->AddPoint(x, y, z, 0);
    z= 50.0;  // 2nd track point
    t=(z-best_par[2])/best_par[5];
    x=best_par[0]+t*best_par[3];
    y=best_par[1]+t*best_par[4];
    track->AddPoint(x, y, z, 1);
  }

  // Draw track hits and track
  for (int p=0; p<3; p++) {
    CRtd_pad_[p]->cd();
    for (int i=0; i<6; i++)
      for (int j=0; j<Nhits_lad[i]; j++)
	if (mm_off[i][j]) mm_off[i][j]->Draw();
    if (TrackFound) {
      for (int i=0; i<Npoints; i++)
	mm[i]->Draw();
      track->Draw();
    }
    CRtd_pad_[p]->Update();
  }

}

//-------------------------------------------------------------------------------------------------

// To comunicate with the "Cosmic Run Track Display",
// we only need the pointers to the three pads: TPad *CRtd_pad_[3]

void ESOccupancyCTClient::CRtd()
{
  Double_t deg=acos(-1.0)/180.0;
  Double_t phi=3.8;              //      sensor tilt in degrees
  Double_t DX=6.3;               //      sensor X projection length (cm)
  Double_t DY=6.3*cos(phi*deg);  //      sensor Y projection length (cm)
  Double_t DZ=39.0;              // all sensors Z projection length (cm)

  TView *view;
  TAxis3D *axis;
  gSystem->Load("libGeom");

  // construct the canvas
  TCanvas *c = new TCanvas("c", "Cosmic Run Track Display", 700,700);
  CRtd_canvas_=c;
  c->Divide(2,1,1e-10,1e-10);
  c->cd(2);
  TPad *p = (TPad*) gPad;
  p->Divide(1,2,1e-10,1e-10);

  if (gGeoManager) delete gGeoManager;
  TGeoManager *gm = new TGeoManager("CR setup", "3D views");
  TGeoMaterial *mat = new TGeoMaterial("Si"); //, 26.98,13,2.7);
  TGeoMedium *med = new TGeoMedium("MED",1,mat);
  TGeoVolume *top = gGeoManager->MakeBox("TOP", med, DX,    DY*5.0/2.0, DZ/2.0); // the box...
  //gGeoManager->SetTopVolume(top);
  gm->SetTopVolume(top);
  TGeoVolume *vol = gGeoManager->MakeBox("BOX", med, DX/2., DY/2.,      0.01);   // a sensor
  vol->SetLineColor(1);
  vol->SetLineWidth(4);

  // create all sensors
  TGeoRotation *sensor_rotation;
  TGeoTranslation *sensor_trans;
  TGeoCombiTrans *sensor_position;
  for (int ladder=0; ladder<6;  ladder++) {
    for (int sensor=0; sensor<10; sensor++) {
      sensor_rotation = new TGeoRotation("",0,phi,0);
      sensor_trans = new TGeoTranslation(-DX/2.0+DX*(sensor%2), -DY*2.0+DY*(sensor/2), -DZ/2.0+ladder*DZ/5.0);
      sensor_position = new TGeoCombiTrans(*sensor_trans, *sensor_rotation);
      top->AddNode(vol,ladder*10+sensor+1, sensor_position);
    }
  }

  // create all strips
  int ndiv=32;
  Double_t start=-DX/2.;
  Double_t step=DX/ndiv;
  TGeoVolume *slice = vol->Divide("SLICE", 1, ndiv, start, step);
  slice->SetLineColor(5);
  slice->SetLineWidth(1);

  top->SetLineColor(kMagenta);
  top->SetVisContainers();
  gGeoManager->SetTopVisible();

  gGeoManager->CloseGeometry();
  gGeoManager->SetNsegments(80);

  // ----- 3D view -----
  c->cd(1);
  CRtd_pad_[0]=(TPad*) gPad;
  top->Draw();
  view = gPad->GetView();
  view->RotateView(-40,90);
  view->ZoomView(0,2);
  //view->ShowAxis();

  // ----- XZ view -----
  p->cd(1);
  CRtd_pad_[1]=(TPad*) gPad;
  top->Draw();
  view = gPad->GetView();
  view->RotateView(-90,90);
  view->SetParallel();
  view->ZoomView(0,1.5);
  view->ShowAxis();
  axis = TAxis3D::GetPadAxis(); // Ask axis pointer
  axis->SetLabelSize(0);
  axis->GetXaxis()->SetTitleColor(2);
  axis->GetXaxis()->SetTitle("strip");
  axis->GetYaxis()->SetTitleColor(3);
  axis->GetYaxis()->SetTitle("sensor");
  axis->GetZaxis()->SetTitleColor(4);
  axis->GetZaxis()->SetTitle("ladder");

  // ----- YZ view -----
  p->cd(2);
  CRtd_pad_[2]=(TPad*) gPad;
  top->Draw();
  view = gPad->GetView();
  view->RotateView(0,90);
  view->SetParallel();
  view->ZoomView(0,1.5);
  view->ShowAxis();
  axis = TAxis3D::GetPadAxis(); // Ask axis pointer
  axis->SetLabelSize(0);
  axis->GetXaxis()->SetTitleColor(2);
  axis->GetXaxis()->SetTitle("strip");
  axis->GetYaxis()->SetTitleColor(3);
  axis->GetYaxis()->SetTitle("sensor");
  axis->GetZaxis()->SetTitleColor(4);
  axis->GetZaxis()->SetTitle("ladder");
}
//-------------------------------------------------------------------------------------------------
