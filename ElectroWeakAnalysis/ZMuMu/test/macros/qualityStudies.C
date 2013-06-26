#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TTree.h"
#include "TAxis.h"
#include "TCanvas.h"
//#if !defined(__CINT__) && !defined(__MAKECINT__)                                                                                                          
#include <string>
#include <iostream>
#include <sstream>
//#endif   

void setGraphics(TH1F *histo){

  histo->SetFillColor(kAzure+7);
  histo->SetLineWidth(2);
  histo->SetLineColor(kBlue+1);
}

TH1F * getTH1Histo(TChain *Events, const char * name,  const string varToPlot,  unsigned int nBins, double fMin, double fMax, TCut cut) {

  TH1F * h = new TH1F(name, name, nBins, fMin, fMax);
  //  Events->Draw("zGoldenMass");
  Events->Project(name, varToPlot.c_str(), cut );
  cout<<"Number of entrie for "<< name << " : "<< h->GetEntries()<<endl;
  setGraphics(h);
  return h;
}



void qualityStudiesZGolden(TFile * output_file, TChain *Events){

  gStyle->SetOptStat();
  gROOT->SetStyle("Plain");
  using namespace std;


  // zGolden plots
  TCut cut_zGolden("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1TrkIso< 3.0 && zGoldenDau2TrkIso < 3.0 && zGoldenDau1Eta<2.1 &&  zGoldenDau2Eta<2.1");
  TDirectory * dir = output_file->mkdir("goodZToMuMuPlots");
  dir->cd();

  TH1F * zMass = getTH1Histo(Events, "zMass", "zGoldenMass", 200, 0, 200,  cut_zGolden) ;
  zMass->Write();
  delete zMass;

  //Quality checks
  // Chi2

  // quality variables
  // caveat: I'm  requiring isolations
  TH1F * dauChi2 = getTH1Histo(Events, "dauChi2", "zGoldenDau1Chi2", 1000, 0, 100,  cut_zGolden) ;
  TH1F * dau2Chi2 = getTH1Histo(Events, "dau2Chi2", "zGoldenDau2Chi2", 1000, 0, 100,  cut_zGolden) ;
  dauChi2->Add(dau2Chi2);
  dauChi2->Write();
  delete dauChi2;
  delete dau2Chi2;
   

  dir->cd();
  TH1F * zMassOneDauChi2Higher10 = getTH1Histo(Events, "zMassOneDauChi2Higher10", "zGoldenMass", 200, 0, 200,  cut_zGolden +"zGoldenDau1Chi2>10 || zGoldenDau2Chi2>10") ;  
  zMassOneDauChi2Higher10->Write();
  delete   zMassOneDauChi2Higher10;

  dir->cd();
  TH1F * zMassBothDauChi2Higher10 = getTH1Histo(Events, "zMassBothDauChi2Higher10", "zGoldenMass", 200, 0, 200,  cut_zGolden +"zGoldenDau1Chi2>10 && zGoldenDau2Chi2>10") ;
  zMassBothDauChi2Higher10->Write();
  delete   zMassBothDauChi2Higher10;

  dir->cd();  
  TH1F *dauChi2NofMuonHits0  = getTH1Histo(Events, "dauChi2NofMuonHits0", "zGoldenDau1Chi2", 1000, 0, 100,  cut_zGolden + "zGoldenDau1NofMuonHits==0") ;
  TH1F *dau2Chi2NofMuonHits0  = getTH1Histo(Events, "dau2Chi2NofMuonHits0", "zGoldenDau2Chi2", 1000, 0, 100,  cut_zGolden + "zGoldenDau2NofMuonHits==0") ;
  dauChi2NofMuonHits0->Add(dau2Chi2NofMuonHits0);
  dauChi2NofMuonHits0->Write();
  delete dauChi2NofMuonHits0;
  delete dau2Chi2NofMuonHits0;


  dir->cd();

  TH1F *dauSaChi2NofSaMuonHits0  = getTH1Histo(Events, "dauSaChi2NofSaMuonHits0", "zGoldenDau1SaChi2", 1000, 0, 100,  cut_zGolden + "zGoldenDau1SaNofMuonHits==0") ;
  TH1F *dau2SaChi2NofSaMuonHits0  = getTH1Histo(Events, "dau2SaChi2NofSaMuonHits0", "zGoldenDau2SaChi2", 1000, 0, 100,  cut_zGolden + "zGoldenDau2SaNofMuonHits==0") ;
  dauSaChi2NofSaMuonHits0->Add(dau2SaChi2NofSaMuonHits0);
  dauSaChi2NofSaMuonHits0->Write();
  delete dauSaChi2NofSaMuonHits0;
  delete dau2SaChi2NofSaMuonHits0;



  TH1F *dauChi2NofStripHits0  = getTH1Histo(Events, "dauChi2NofStripHits0", "zGoldenDau1Chi2", 1000, 0, 100,  cut_zGolden + "zGoldenDau1NofStripHits<10") ;
  TH1F *dau2Chi2NofStripHits0  = getTH1Histo(Events, "dau2Chi2NofStripHits0", "zGoldenDau2Chi2", 1000, 0, 100,  cut_zGolden + "zGoldenDau2NofStripHits<10") ;
  dauChi2NofStripHits0->Add(dau2Chi2NofStripHits0);
  dauChi2NofStripHits0->Write();
  //cout<<"Number of zCandidate with at least one daughter with Chi2 higher: "<<zMass->GetEntries()<<endl;
  delete dauChi2NofStripHits0;
  delete dau2Chi2NofStripHits0;
  
  
 

 // Number of  Strips Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dauNofStripHits  = getTH1Histo(Events, "dauNofStripHits", "zGoldenDau1NofStripHits", 100, 0, 100,  cut_zGolden ) ;
  TH1F *dau2NofStripHits  = getTH1Histo(Events, "dau2NofStripHits", "zGoldenDau2NofStripHits", 100, 0, 100,  cut_zGolden ) ;
  dauNofStripHits->Add(dau2NofStripHits);
  dauNofStripHits->Write();
  delete dauNofStripHits;
  delete dau2NofStripHits;


  dir->cd();
  TH1F * zMassBothDauNofStripsHitsLower10 = getTH1Histo(Events, "zMassBothDauNofStripsHitsLower10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1NofStripHits<10 && zGoldenDau2NofStripHits<10" );
  zMassBothDauNofStripsHitsLower10->Write();
  delete zMassBothDauNofStripsHitsLower10;

  dir->cd();
  TH1F * zMassOneDauNofStripsHitsLower10 = getTH1Histo(Events, "zMassOneDauNofStripsHitsLower10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1NofStripHits<10 || zGoldenDau2NofStripHits<10" );
  zMassOneDauNofStripsHitsLower10->Write();
  delete zMassOneDauNofStripsHitsLower10;

  

// Number of  Strips Hits for inner track
  dir->cd();
  // caveat: I'm  requiring isolations
   
  TH1F *dauTrkNofStripHits  = getTH1Histo(Events, "dauTrkNofStripHits", "zGoldenDau1TrkNofStripHits", 100, 0, 100,  cut_zGolden ) ;
  TH1F *dau2TrkNofStripHits  = getTH1Histo(Events, "dau2TrkNofStripHits", "zGoldenDau2TrkNofStripHits", 100, 0, 100,  cut_zGolden ) ;
  dauTrkNofStripHits->Add(dau2TrkNofStripHits);
  dauTrkNofStripHits->Write();
  delete dauTrkNofStripHits;
  delete dau2TrkNofStripHits;


  dir->cd();
  TH1F * zMassBothDauTrkNofStripsHitsLower10 = getTH1Histo(Events, "zMassBothDauTrkNofStripsHitsLower10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1TrkNofStripHits<10 && zGoldenDau2TrkNofStripHits<10" );
  zMassBothDauTrkNofStripsHitsLower10->Write();
  delete zMassBothDauTrkNofStripsHitsLower10;

  dir->cd();
  TH1F * zMassOneDauTrkNofStripsHitsLower10 = getTH1Histo(Events, "zMassOneDauTrkNofStripsHitsLower10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1TrkNofStripHits<10 || zGoldenDau2TrkNofStripHits<10" );
  zMassOneDauTrkNofStripsHitsLower10->Write();
  delete zMassOneDauTrkNofStripsHitsLower10;

  


 // Number of  Pixel Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dauNofPixelHits  = getTH1Histo(Events, "dauNofPixelHits", "zGoldenDau1NofPixelHits", 100, 0, 100,  cut_zGolden ) ;
  TH1F *dau2NofPixelHits  = getTH1Histo(Events, "dau2NofPixelHits", "zGoldenDau2NofPixelHits", 100, 0, 100,  cut_zGolden ) ;
  dauNofPixelHits->Add(dau2NofPixelHits);
  dauNofPixelHits->Write();
  delete dauNofPixelHits;
  delete dau2NofPixelHits;


  dir->cd();
  TH1F * zMassBothDauNofPixelsHitsLower10 = getTH1Histo(Events, "zMassBothDauNofPixelsHitsLower10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1NofPixelHits<10 && zGoldenDau2NofPixelHits<10" );
  zMassBothDauNofPixelsHitsLower10->Write();
  delete zMassBothDauNofPixelsHitsLower10;

  dir->cd();
  TH1F * zMassOneDauNofPixelsHitsLower10 = getTH1Histo(Events, "zMassOneDauNofPixelsHitsLower10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1NofPixelHits<10 || zGoldenDau2NofPixelHits<10" );
  zMassOneDauNofPixelsHitsLower10->Write();
  delete zMassOneDauNofPixelsHitsLower10;

  


 // Number of  Pixel Hits for inner track
 dir->cd();
  // caveat: I'm  requiring isolations
   
  TH1F *dauTrkNofPixelHits  = getTH1Histo(Events, "dauTrkNofPixelHits", "zGoldenDau1TrkNofPixelHits", 100, 0, 100,  cut_zGolden ) ;
  TH1F *dau2TrkNofPixelHits  = getTH1Histo(Events, "dau2TrkNofPixelHits", "zGoldenDau2TrkNofPixelHits", 100, 0, 100,  cut_zGolden ) ;
  dauTrkNofPixelHits->Add(dau2TrkNofPixelHits);
  dauTrkNofPixelHits->Write();
  delete dauTrkNofPixelHits;
  delete dau2TrkNofPixelHits;


  dir->cd();
  TH1F * zMassBothDauTrkNofPixelsHitsLower1 = getTH1Histo(Events, "zMassBothDauTrkNofPixelsHitsLower1","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1TrkNofPixelHits<1 && zGoldenDau2TrkNofPixelHits<1" );
  zMassBothDauTrkNofPixelsHitsLower1->Write();
  delete zMassBothDauTrkNofPixelsHitsLower1;

  dir->cd();
  TH1F * zMassOneDauTrkNofPixelsHitsLower1 = getTH1Histo(Events, "zMassOneDauTrkNofPixelsHitsLower1","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1TrkNofPixelHits<1 || zGoldenDau2TrkNofPixelHits<1" );
  zMassOneDauTrkNofPixelsHitsLower1->Write();
  delete zMassOneDauTrkNofPixelsHitsLower1;
  






  // Number of  Muon Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dauNofMuonHits  = getTH1Histo(Events, "dauNofMuonHits", "zGoldenDau1NofMuonHits", 100, 0, 100,  cut_zGolden ) ;
  TH1F *dau2NofMuonHits  = getTH1Histo(Events, "dau2NofMuonHits", "zGoldenDau2NofMuonHits", 100, 0, 100,  cut_zGolden ) ;
  dauNofMuonHits->Add(dau2NofMuonHits);
  dauNofMuonHits->Write();
  delete dauNofMuonHits;
  delete dau2NofMuonHits;


  dir->cd();
  TH1F * zMassBothDauNofMuonsHitsLower1 = getTH1Histo(Events, "zMassBothDauNofMuonsHitsLower1","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1NofMuonHits<1 && zGoldenDau2NofMuonHits<1" );
  zMassBothDauNofMuonsHitsLower1->Write();
  delete zMassBothDauNofMuonsHitsLower1;

  dir->cd();
  TH1F * zMassOneDauNofMuonsHitsLower1 = getTH1Histo(Events, "zMassOneDauNofMuonsHitsLowe10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1NofMuonHits<1 || zGoldenDau2NofMuonHits<1" );
  zMassOneDauNofMuonsHitsLower1->Write();
  delete zMassOneDauNofMuonsHitsLower1;

  

  // Number of  Muon Hits for outer track
 dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dauSaNofMuonHits  = getTH1Histo(Events, "dauSaNofMuonHits", "zGoldenDau1SaNofMuonHits", 100, 0, 100,  cut_zGolden ) ;
  TH1F *dau2SaNofMuonHits  = getTH1Histo(Events, "dau2SaNofMuonHits", "zGoldenDau2SaNofMuonHits", 100, 0, 100,  cut_zGolden ) ;
  dauSaNofMuonHits->Add(dau2SaNofMuonHits);
  dauSaNofMuonHits->Write();
  delete dauSaNofMuonHits;
  delete dau2SaNofMuonHits;


  dir->cd();
  TH1F * zMassBothDauSaNofMuonsHitsLower1 = getTH1Histo(Events, "zMassBothDauSaNofMuonsHitsLower1","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1SaNofMuonHits<1 && zGoldenDau2SaNofMuonHits<1" );
  zMassBothDauSaNofMuonsHitsLower1->Write();
  delete zMassBothDauSaNofMuonsHitsLower1;

  dir->cd();
  TH1F * zMassOneDauSaNofMuonsHitsLower1 = getTH1Histo(Events, "zMassOneDauSaNofMuonsHitsLowe10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1SaNofMuonHits<1 || zGoldenDau2SaNofMuonHits<1" );
  zMassOneDauSaNofMuonsHitsLower1->Write();
  delete zMassOneDauSaNofMuonsHitsLower1;
 
 



  // dxyFromBS
  dir->cd();

  TH1F *dauDxy  = getTH1Histo(Events, "dauDxy", "zGoldenDau1dxyFromBS", 200, -1, 1,  cut_zGolden ) ;
  TH1F *dau2Dxy  = getTH1Histo(Events, "dau2Dxy", "zGoldenDau2dxyFromBS", 200, -1, 1,  cut_zGolden ) ;
  dauDxy->Add(dau2Dxy);
  dauDxy->Write();
  delete dauDxy;
  delete dau2Dxy;


  TH1F * zMassBothDaudxyFromBSHigher0_2  = getTH1Histo(Events, "zMassBothDaudxyFromBSHigher0_2", "zGoldenMass", 200, -1, 1,  cut_zGolden + "zGoldenDau1dxyFromBS>0.2 && zGoldenDau2dxyFromBS>0.2" ) ;
  zMassBothDaudxyFromBSHigher0_2->Write();
  delete zMassBothDaudxyFromBSHigher0_2;



  dir->cd();
  TH1F * zMassOneDaudxyFromBSHigher0_2  = getTH1Histo(Events, "zMassOneDaudxyFromBSHigher0_2", "zGoldenMass", 200, -1, 1,  cut_zGolden + "zGoldenDau1dxyFromBS>0.2 || zGoldenDau2dxyFromBS>0.2" ) ;
  zMassOneDaudxyFromBSHigher0_2->Write();
  delete zMassOneDaudxyFromBSHigher0_2;
 

  // isTrackerMuon
  dir->cd(); 
  TH1F * zMassBothDauNoTrackerMuon    = getTH1Histo(Events, "zMassBothDauNoTrackerMuon", "zGoldenMass", 200, 0, 200,  cut_zGolden + "zGoldenDau1TrackerMuonBit==0 && zGoldenDau2TrackerMuonBit==0");
  zMassBothDauNoTrackerMuon->Write();
  delete zMassBothDauNoTrackerMuon;

  dir->cd(); 
  TH1F * zMassOneDauNoTrackerMuon    = getTH1Histo(Events, "zMassOneDauNoTrackerMuon", "zGoldenMass", 200, 0, 200,  cut_zGolden + "zGoldenDau1TrackerMuonBit==0 || zGoldenDau2TrackerMuonBit==0");
  zMassOneDauNoTrackerMuon->Write();
  delete zMassOneDauNoTrackerMuon;
  


  // Eta distribution if MuonHits is zero
  
  dir->cd();
  TH1F * dauEtaNofMuonHits0 = getTH1Histo(Events, "dauEtaNofMuonHits0", "zGoldenDau1Eta", 200, -5, 5,  cut_zGolden +  "zGoldenDau1NofMuonHits==0" );
  TH1F * dau2EtaNofMuonHits0 = getTH1Histo(Events, "dau2EtaNofMuonHits0", "zGoldenDau2Eta", 200, -5, 5,  cut_zGolden +  "zGoldenDau2NofMuonHits==0" );
  dauEtaNofMuonHits0->Add(dau2EtaNofMuonHits0);
  dauEtaNofMuonHits0->Write();
  delete dauEtaNofMuonHits0;
  delete dau2EtaNofMuonHits0;
  

  // Correlation study

  // *** Chi2 vs MuonHits  ***
 
  dir->cd();
  TH2F * Chi2VsMuonHits = new TH2F("Chi2VsMuonHits", "Chi2VsMuonHits", 100, 0, 60, 100, 0, 6);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events->Project("Chi2VsMuonHits", "zGoldenDau1Chi2:zGoldenDau1NofMuonHits", cut_zGolden);         
  Chi2VsMuonHits->SetDrawOption("Box"); 

  Chi2VsMuonHits->Write();

  delete Chi2VsMuonHits;


  // *** Chi2 vs StripHits  ***
 
  dir->cd();
  TH2F * Chi2VsStripHits = new TH2F("Chi2VsStripHits", "Chi2VsStripHits", 100, 0, 30, 100, 0, 6);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events->Project("Chi2VsStripHits", "zGoldenDau1Chi2:zGoldenDau1NofStripHits", cut_zGolden);         
  Chi2VsStripHits->SetDrawOption("Box"); 

  Chi2VsStripHits->Write();

  delete Chi2VsStripHits;

   // *** MuonHits vs Eta  ***
 
  dir->cd();
  TH2F * MuonHitsVsEta = new TH2F("MuonHitsVsEta", "MuonHitsVsEta", 100, -2.5, 2.5, 100, 0, 60);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events->Project("MuonHitsVsEta", "zGoldenDau1NofMuonHits:zGoldenDau1Eta", cut_zGolden);         
  MuonHitsVsEta->SetDrawOption("Box"); 

  MuonHitsVsEta->Write();

  delete MuonHitsVsEta;


  output_file->cd("/");

  
 

  //   output_file->Close(); 
 
}


void qualityStudiesZMuSta(TFile * output_file, TChain *Events){

  gStyle->SetOptStat();
  gROOT->SetStyle("Plain");
  using namespace std;


  // zzMuSta plots
  TCut cut_zMuSta("zMuStaMass>60 && zMuStaMass<120 && zMuStaDau1Pt> 20 && zMuStaDau2Pt>20 && zMuStaDau1TrkIso< 3.0 && zMuStaDau2TrkIso < 3.0 && zMuStaDau1Eta<2.1 &&  zMuStaDau2Eta<2.1");
  TDirectory * dir = output_file->mkdir("goodZToMuMuOneStandAloneMuonPlots");
  dir->cd();

  TH1F * zMass = getTH1Histo(Events, "zMass", "zMuStaMass", 200, 0, 200,  cut_zMuSta) ;
  zMass->Write();
  delete zMass;

  //Quality checks
  // Chi2

  // quality variables
  // caveat: I'm  requiring isolations
  TH1F * dauChi2 = getTH1Histo(Events, "dauChi2", "zMuStaDau1Chi2", 1000, 0, 100,  cut_zMuSta) ;
  TH1F * dau2Chi2 = getTH1Histo(Events, "dau2Chi2", "zMuStaDau2Chi2", 1000, 0, 100,  cut_zMuSta) ;
  dauChi2->Add(dau2Chi2);
  dauChi2->Write();
  delete dauChi2;
  delete dau2Chi2;
   

  dir->cd();
  TH1F * zMassOneDauChi2Higher10 = getTH1Histo(Events, "zMassOneDauChi2Higher10", "zMuStaMass", 200, 0, 200,  cut_zMuSta +"zMuStaDau1Chi2>10 || zMuStaDau2Chi2>10") ;  
  zMassOneDauChi2Higher10->Write();
  delete   zMassOneDauChi2Higher10;

  dir->cd();
  TH1F * zMassBothDauChi2Higher10 = getTH1Histo(Events, "zMassBothDauChi2Higher10", "zMuStaMass", 200, 0, 200,  cut_zMuSta +"zMuStaDau1Chi2>10 && zMuStaDau2Chi2>10") ;
  zMassBothDauChi2Higher10->Write();
  delete   zMassBothDauChi2Higher10;

  dir->cd();  
  TH1F *dauChi2NofMuonHits0  = getTH1Histo(Events, "dauChi2NofMuonHits0", "zMuStaDau1Chi2", 1000, 0, 100,  cut_zMuSta + "zMuStaDau1NofMuonHits==0") ;
  TH1F *dau2Chi2NofMuonHits0  = getTH1Histo(Events, "dau2Chi2NofMuonHits0", "zMuStaDau2Chi2", 1000, 0, 100,  cut_zMuSta + "zMuStaDau2NofMuonHits==0") ;
  dauChi2NofMuonHits0->Add(dau2Chi2NofMuonHits0);
  dauChi2NofMuonHits0->Write();
  delete dauChi2NofMuonHits0;
  delete dau2Chi2NofMuonHits0;


  dir->cd();

  TH1F *dauSaChi2NofSaMuonHits0  = getTH1Histo(Events, "dauSaChi2NofSaMuonHits0", "zMuStaDau1SaChi2", 1000, 0, 100,  cut_zMuSta + "zMuStaDau1SaNofMuonHits==0") ;
  TH1F *dau2SaChi2NofSaMuonHits0  = getTH1Histo(Events, "dau2SaChi2NofSaMuonHits0", "zMuStaDau2SaChi2", 1000, 0, 100,  cut_zMuSta + "zMuStaDau2SaNofMuonHits==0") ;
  dauSaChi2NofSaMuonHits0->Add(dau2SaChi2NofSaMuonHits0);
  dauSaChi2NofSaMuonHits0->Write();
  delete dauSaChi2NofSaMuonHits0;
  delete dau2SaChi2NofSaMuonHits0;



  TH1F *dauChi2NofStripHits0  = getTH1Histo(Events, "dauChi2NofStripHits0", "zMuStaDau1Chi2", 1000, 0, 100,  cut_zMuSta + "zMuStaDau1NofStripHits<10") ;
  TH1F *dau2Chi2NofStripHits0  = getTH1Histo(Events, "dau2Chi2NofStripHits0", "zMuStaDau2Chi2", 1000, 0, 100,  cut_zMuSta + "zMuStaDau2NofStripHits<10") ;
  dauChi2NofStripHits0->Add(dau2Chi2NofStripHits0);
  dauChi2NofStripHits0->Write();
  //cout<<"Number of zCandidate with at least one daughter with Chi2 higher: "<<zMass->GetEntries()<<endl;
  delete dauChi2NofStripHits0;
  delete dau2Chi2NofStripHits0;
  
  
 

 // Number of  Strips Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dauGlbNofStripHits  = getTH1Histo(Events, "dauGlbNofStripHits", "zMuStaDau1NofStripHits", 100, 0, 100,  cut_zMuSta && "zMuStaDau1GlobalMuonBit==1" ) ;
  TH1F *dau2NofStripHits  = getTH1Histo(Events, "dau2NofStripHits", "zMuStaDau2NofStripHits", 100, 0, 100,  cut_zMuSta && "zMuStaDau2GlobalMuonBit==1") ;
  dauGlbNofStripHits->Add(dau2NofStripHits);
  dauGlbNofStripHits->Write();
  delete dauGlbNofStripHits;
  delete dau2NofStripHits;
  






  // Number of  Muon Hits for global 
  dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dauGlbNofMuonHits  = getTH1Histo(Events, "dauGlbNofMuonHits", "zMuStaDau1NofMuonHits", 100, 0, 100,  cut_zMuSta && "zMuStaDau1GlobalMuonBit==1") ;
  TH1F *dau2NofMuonHits  = getTH1Histo(Events, "dau2NofMuonHits", "zMuStaDau2NofMuonHits", 100, 0, 100,  cut_zMuSta && "zMuStaDau2GlobalMuonBit==1") ;
  dauGlbNofMuonHits->Add(dau2NofMuonHits);
  dauGlbNofMuonHits->Write();
  delete dauGlbNofMuonHits;
  delete dau2NofMuonHits;


  dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dauStaNofMuonHits  = getTH1Histo(Events, "dauStaNofMuonHits", "zMuStaDau1SaNofMuonHits", 100, 0, 100,  cut_zMuSta && "zMuStaDau1StandAloneBit==1" && "zMuStaDau1GlobalMuonBit==0") ;
  TH1F *dau2NofMuonHits  = getTH1Histo(Events, "dau2NofMuonHits", "zMuStaDau2SaNofMuonHits", 100, 0, 100,  cut_zMuSta && "zMuStaDau2StandAloneBit==1" && "zMuStaDau2GlobalMuonBit==0" ) ;
  dauStaNofMuonHits->Add(dau2NofMuonHits);
  dauStaNofMuonHits->Write();
  delete dauStaNofMuonHits;
  delete dau2NofMuonHits;

 
 



  // dxyFromBS
  dir->cd();

  TH1F *dauDxy  = getTH1Histo(Events, "dauDxy", "zMuStaDau1dxyFromBS", 200, -1, 1,  cut_zMuSta ) ;
  TH1F *dau2Dxy  = getTH1Histo(Events, "dau2Dxy", "zMuStaDau2dxyFromBS", 200, -1, 1,  cut_zMuSta ) ;
  dauDxy->Add(dau2Dxy);
  dauDxy->Write();
  delete dauDxy;
  delete dau2Dxy;


  TH1F * zMassBothDaudxyFromBSHigher0_2  = getTH1Histo(Events, "zMassBothDaudxyFromBSHigher0_2", "zMuStaMass", 200, -1, 1,  cut_zMuSta + "zMuStaDau1dxyFromBS>0.2 && zMuStaDau2dxyFromBS>0.2" ) ;
  zMassBothDaudxyFromBSHigher0_2->Write();
  delete zMassBothDaudxyFromBSHigher0_2;



  dir->cd();
  TH1F * zMassOneDaudxyFromBSHigher0_2  = getTH1Histo(Events, "zMassOneDaudxyFromBSHigher0_2", "zMuStaMass", 200, -1, 1,  cut_zMuSta + "zMuStaDau1dxyFromBS>0.2 || zMuStaDau2dxyFromBS>0.2" ) ;
  zMassOneDaudxyFromBSHigher0_2->Write();
  delete zMassOneDaudxyFromBSHigher0_2;
 


  // Eta distribution if MuonHits is zero
  
  dir->cd();
  TH1F * dauEtaNofMuonHits0 = getTH1Histo(Events, "dauEtaNofMuonHits0", "zMuStaDau1Eta", 200, -5, 5,  cut_zMuSta +  "zMuStaDau1NofMuonHits==0" );
  TH1F * dau2EtaNofMuonHits0 = getTH1Histo(Events, "dau2EtaNofMuonHits0", "zMuStaDau2Eta", 200, -5, 5,  cut_zMuSta +  "zMuStaDau2NofMuonHits==0" );
  dauEtaNofMuonHits0->Add(dau2EtaNofMuonHits0);
  dauEtaNofMuonHits0->Write();
  delete dauEtaNofMuonHits0;
  delete dau2EtaNofMuonHits0;
  

  // Correlation study

  // *** Chi2 vs MuonHits  ***
 
  dir->cd();
  TH2F * Chi2VsMuonHits = new TH2F("Chi2VsMuonHits", "Chi2VsMuonHits", 100, 0, 60, 100, 0, 6);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events->Project("Chi2VsMuonHits", "zMuStaDau1Chi2:zMuStaDau1NofMuonHits", cut_zMuSta);         
  Chi2VsMuonHits->SetDrawOption("Box"); 

  Chi2VsMuonHits->Write();

  delete Chi2VsMuonHits;


  // *** Chi2 vs StripHits  ***
 
  dir->cd();
  TH2F * Chi2VsStripHits = new TH2F("Chi2VsStripHits", "Chi2VsStripHits", 100, 0, 30, 100, 0, 6);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events->Project("Chi2VsStripHits", "zMuStaDau1Chi2:zMuStaDau1NofStripHits", cut_zMuSta);         
  Chi2VsStripHits->SetDrawOption("Box"); 

  Chi2VsStripHits->Write();

  delete Chi2VsStripHits;

   // *** MuonHits vs Eta  ***
 
  dir->cd();
  TH2F * MuonHitsVsEta = new TH2F("MuonHitsVsEta", "MuonHitsVsEta", 100, -2.5, 2.5, 100, 0, 60);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events->Project("MuonHitsVsEta", "zMuStaDau1NofMuonHits:zMuStaDau1Eta", cut_zMuSta);         
  MuonHitsVsEta->SetDrawOption("Box"); 

  MuonHitsVsEta->Write();

  delete MuonHitsVsEta;


  output_file->cd("/");

  
 

  //   output_file->Close(); 
 
}




void qualityStudiesZGoldenNotIso(TFile * output_file, TChain *Events){

  gStyle->SetOptStat();
  gROOT->SetStyle("Plain");
  using namespace std;


  // zGolden plots
  TCut cut_zGolden("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && (zGoldenDau1TrkIso> 3.0 || zGoldenDau2TrkIso > 3.0 ) && zGoldenDau1Eta<2.1 &&  zGoldenDau2Eta<2.1");
  TDirectory * dir = output_file->mkdir("nonIsolatedZToMuMuPlots");
  dir->cd();

  TH1F * zMass = getTH1Histo(Events, "zMass", "zGoldenMass", 200, 0, 200,  cut_zGolden) ;
  zMass->Write();
  delete zMass;

  //Quality checks
  // Chi2

  // quality variables
  // caveat: I'm  requiring isolations
  TH1F * dauChi2 = getTH1Histo(Events, "dauChi2", "zGoldenDau1Chi2", 1000, 0, 100,  cut_zGolden) ;
  TH1F * dau2Chi2 = getTH1Histo(Events, "dau2Chi2", "zGoldenDau2Chi2", 1000, 0, 100,  cut_zGolden) ;
  dauChi2->Add(dau2Chi2);
  dauChi2->Write();
  delete dauChi2;
  delete dau2Chi2;
   

  dir->cd();
  TH1F * zMassOneDauChi2Higher10 = getTH1Histo(Events, "zMassOneDauChi2Higher10", "zGoldenMass", 200, 0, 200,  cut_zGolden +"zGoldenDau1Chi2>10 || zGoldenDau2Chi2>10") ;  
  zMassOneDauChi2Higher10->Write();
  delete   zMassOneDauChi2Higher10;

  dir->cd();
  TH1F * zMassBothDauChi2Higher10 = getTH1Histo(Events, "zMassBothDauChi2Higher10", "zGoldenMass", 200, 0, 200,  cut_zGolden +"zGoldenDau1Chi2>10 && zGoldenDau2Chi2>10") ;
  zMassBothDauChi2Higher10->Write();
  delete   zMassBothDauChi2Higher10;

  dir->cd();  
  TH1F *dauChi2NofMuonHits0  = getTH1Histo(Events, "dauChi2NofMuonHits0", "zGoldenDau1Chi2", 1000, 0, 100,  cut_zGolden + "zGoldenDau1NofMuonHits==0") ;
  TH1F *dau2Chi2NofMuonHits0  = getTH1Histo(Events, "dau2Chi2NofMuonHits0", "zGoldenDau2Chi2", 1000, 0, 100,  cut_zGolden + "zGoldenDau2NofMuonHits==0") ;
  dauChi2NofMuonHits0->Add(dau2Chi2NofMuonHits0);
  dauChi2NofMuonHits0->Write();
  delete dauChi2NofMuonHits0;
  delete dau2Chi2NofMuonHits0;


  dir->cd();

  TH1F *dauSaChi2NofSaMuonHits0  = getTH1Histo(Events, "dauSaChi2NofSaMuonHits0", "zGoldenDau1SaChi2", 1000, 0, 100,  cut_zGolden + "zGoldenDau1SaNofMuonHits==0") ;
  TH1F *dau2SaChi2NofSaMuonHits0  = getTH1Histo(Events, "dau2SaChi2NofSaMuonHits0", "zGoldenDau2SaChi2", 1000, 0, 100,  cut_zGolden + "zGoldenDau2SaNofMuonHits==0") ;
  dauSaChi2NofSaMuonHits0->Add(dau2SaChi2NofSaMuonHits0);
  dauSaChi2NofSaMuonHits0->Write();
  delete dauSaChi2NofSaMuonHits0;
  delete dau2SaChi2NofSaMuonHits0;



  TH1F *dauChi2NofStripHits0  = getTH1Histo(Events, "dauChi2NofStripHits0", "zGoldenDau1Chi2", 1000, 0, 100,  cut_zGolden + "zGoldenDau1NofStripHits<10") ;
  TH1F *dau2Chi2NofStripHits0  = getTH1Histo(Events, "dau2Chi2NofStripHits0", "zGoldenDau2Chi2", 1000, 0, 100,  cut_zGolden + "zGoldenDau2NofStripHits<10") ;
  dauChi2NofStripHits0->Add(dau2Chi2NofStripHits0);
  dauChi2NofStripHits0->Write();
  //cout<<"Number of zCandidate with at least one daughter with Chi2 higher: "<<zMass->GetEntries()<<endl;
  delete dauChi2NofStripHits0;
  delete dau2Chi2NofStripHits0;
  
  
 

 // Number of  Strips Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dauNofStripHits  = getTH1Histo(Events, "dauNofStripHits", "zGoldenDau1NofStripHits", 100, 0, 100,  cut_zGolden ) ;
  TH1F *dau2NofStripHits  = getTH1Histo(Events, "dau2NofStripHits", "zGoldenDau2NofStripHits", 100, 0, 100,  cut_zGolden ) ;
  dauNofStripHits->Add(dau2NofStripHits);
  dauNofStripHits->Write();
  delete dauNofStripHits;
  delete dau2NofStripHits;


  dir->cd();
  TH1F * zMassBothDauNofStripsHitsLower10 = getTH1Histo(Events, "zMassBothDauNofStripsHitsLower10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1NofStripHits<10 && zGoldenDau2NofStripHits<10" );
  zMassBothDauNofStripsHitsLower10->Write();
  delete zMassBothDauNofStripsHitsLower10;

  dir->cd();
  TH1F * zMassOneDauNofStripsHitsLower10 = getTH1Histo(Events, "zMassOneDauNofStripsHitsLower10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1NofStripHits<10 || zGoldenDau2NofStripHits<10" );
  zMassOneDauNofStripsHitsLower10->Write();
  delete zMassOneDauNofStripsHitsLower10;

  

// Number of  Strips Hits for inner track
  dir->cd();
  // caveat: I'm  requiring isolations
   
  TH1F *dauTrkNofStripHits  = getTH1Histo(Events, "dauTrkNofStripHits", "zGoldenDau1TrkNofStripHits", 100, 0, 100,  cut_zGolden ) ;
  TH1F *dau2TrkNofStripHits  = getTH1Histo(Events, "dau2TrkNofStripHits", "zGoldenDau2TrkNofStripHits", 100, 0, 100,  cut_zGolden ) ;
  dauTrkNofStripHits->Add(dau2TrkNofStripHits);
  dauTrkNofStripHits->Write();
  delete dauTrkNofStripHits;
  delete dau2TrkNofStripHits;


  dir->cd();
  TH1F * zMassBothDauTrkNofStripsHitsLower10 = getTH1Histo(Events, "zMassBothDauTrkNofStripsHitsLower10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1TrkNofStripHits<10 && zGoldenDau2TrkNofStripHits<10" );
  zMassBothDauTrkNofStripsHitsLower10->Write();
  delete zMassBothDauTrkNofStripsHitsLower10;

  dir->cd();
  TH1F * zMassOneDauTrkNofStripsHitsLower10 = getTH1Histo(Events, "zMassOneDauTrkNofStripsHitsLower10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1TrkNofStripHits<10 || zGoldenDau2TrkNofStripHits<10" );
  zMassOneDauTrkNofStripsHitsLower10->Write();
  delete zMassOneDauTrkNofStripsHitsLower10;

  


 // Number of  Pixel Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dauNofPixelHits  = getTH1Histo(Events, "dauNofPixelHits", "zGoldenDau1NofPixelHits", 100, 0, 100,  cut_zGolden ) ;
  TH1F *dau2NofPixelHits  = getTH1Histo(Events, "dau2NofPixelHits", "zGoldenDau2NofPixelHits", 100, 0, 100,  cut_zGolden ) ;
  dauNofPixelHits->Add(dau2NofPixelHits);
  dauNofPixelHits->Write();
  delete dauNofPixelHits;
  delete dau2NofPixelHits;


  dir->cd();
  TH1F * zMassBothDauNofPixelsHitsLower10 = getTH1Histo(Events, "zMassBothDauNofPixelsHitsLower10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1NofPixelHits<10 && zGoldenDau2NofPixelHits<10" );
  zMassBothDauNofPixelsHitsLower10->Write();
  delete zMassBothDauNofPixelsHitsLower10;

  dir->cd();
  TH1F * zMassOneDauNofPixelsHitsLower10 = getTH1Histo(Events, "zMassOneDauNofPixelsHitsLower10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1NofPixelHits<10 || zGoldenDau2NofPixelHits<10" );
  zMassOneDauNofPixelsHitsLower10->Write();
  delete zMassOneDauNofPixelsHitsLower10;

  


 // Number of  Pixel Hits for inner track
 dir->cd();
  // caveat: I'm  requiring isolations
   
  TH1F *dauTrkNofPixelHits  = getTH1Histo(Events, "dauTrkNofPixelHits", "zGoldenDau1TrkNofPixelHits", 100, 0, 100,  cut_zGolden ) ;
  TH1F *dau2TrkNofPixelHits  = getTH1Histo(Events, "dau2TrkNofPixelHits", "zGoldenDau2TrkNofPixelHits", 100, 0, 100,  cut_zGolden ) ;
  dauTrkNofPixelHits->Add(dau2TrkNofPixelHits);
  dauTrkNofPixelHits->Write();
  delete dauTrkNofPixelHits;
  delete dau2TrkNofPixelHits;


  dir->cd();
  TH1F * zMassBothDauTrkNofPixelsHitsLower1 = getTH1Histo(Events, "zMassBothDauTrkNofPixelsHitsLower1","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1TrkNofPixelHits<1 && zGoldenDau2TrkNofPixelHits<1" );
  zMassBothDauTrkNofPixelsHitsLower1->Write();
  delete zMassBothDauTrkNofPixelsHitsLower1;

  dir->cd();
  TH1F * zMassOneDauTrkNofPixelsHitsLower1 = getTH1Histo(Events, "zMassOneDauTrkNofPixelsHitsLower1","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1TrkNofPixelHits<1 || zGoldenDau2TrkNofPixelHits<1" );
  zMassOneDauTrkNofPixelsHitsLower1->Write();
  delete zMassOneDauTrkNofPixelsHitsLower1;
  






  // Number of  Muon Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dauNofMuonHits  = getTH1Histo(Events, "dauNofMuonHits", "zGoldenDau1NofMuonHits", 100, 0, 100,  cut_zGolden ) ;
  TH1F *dau2NofMuonHits  = getTH1Histo(Events, "dau2NofMuonHits", "zGoldenDau2NofMuonHits", 100, 0, 100,  cut_zGolden ) ;
  dauNofMuonHits->Add(dau2NofMuonHits);
  dauNofMuonHits->Write();
  delete dauNofMuonHits;
  delete dau2NofMuonHits;


  dir->cd();
  TH1F * zMassBothDauNofMuonsHitsLower1 = getTH1Histo(Events, "zMassBothDauNofMuonsHitsLower1","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1NofMuonHits<1 && zGoldenDau2NofMuonHits<1" );
  zMassBothDauNofMuonsHitsLower1->Write();
  delete zMassBothDauNofMuonsHitsLower1;

  dir->cd();
  TH1F * zMassOneDauNofMuonsHitsLower1 = getTH1Histo(Events, "zMassOneDauNofMuonsHitsLowe10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1NofMuonHits<1 || zGoldenDau2NofMuonHits<1" );
  zMassOneDauNofMuonsHitsLower1->Write();
  delete zMassOneDauNofMuonsHitsLower1;

  

  // Number of  Muon Hits for outer track
 dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dauSaNofMuonHits  = getTH1Histo(Events, "dauSaNofMuonHits", "zGoldenDau1SaNofMuonHits", 100, 0, 100,  cut_zGolden ) ;
  TH1F *dau2SaNofMuonHits  = getTH1Histo(Events, "dau2SaNofMuonHits", "zGoldenDau2SaNofMuonHits", 100, 0, 100,  cut_zGolden ) ;
  dauSaNofMuonHits->Add(dau2SaNofMuonHits);
  dauSaNofMuonHits->Write();
  delete dauSaNofMuonHits;
  delete dau2SaNofMuonHits;


  dir->cd();
  TH1F * zMassBothDauSaNofMuonsHitsLower1 = getTH1Histo(Events, "zMassBothDauSaNofMuonsHitsLower1","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1SaNofMuonHits<1 && zGoldenDau2SaNofMuonHits<1" );
  zMassBothDauSaNofMuonsHitsLower1->Write();
  delete zMassBothDauSaNofMuonsHitsLower1;

  dir->cd();
  TH1F * zMassOneDauSaNofMuonsHitsLower1 = getTH1Histo(Events, "zMassOneDauSaNofMuonsHitsLowe10","zGoldenMass", 200, 0, 200,cut_zGolden +"zGoldenDau1SaNofMuonHits<1 || zGoldenDau2SaNofMuonHits<1" );
  zMassOneDauSaNofMuonsHitsLower1->Write();
  delete zMassOneDauSaNofMuonsHitsLower1;
 
 



  // dxyFromBS
  dir->cd();

  TH1F *dauDxy  = getTH1Histo(Events, "dauDxy", "zGoldenDau1dxyFromBS", 200, -1, 1,  cut_zGolden ) ;
  TH1F *dau2Dxy  = getTH1Histo(Events, "dau2Dxy", "zGoldenDau2dxyFromBS", 200, -1, 1,  cut_zGolden ) ;
  dauDxy->Add(dau2Dxy);
  dauDxy->Write();
  delete dauDxy;
  delete dau2Dxy;


  TH1F * zMassBothDaudxyFromBSHigher0_2  = getTH1Histo(Events, "zMassBothDaudxyFromBSHigher0_2", "zGoldenMass", 200, -1, 1,  cut_zGolden + "zGoldenDau1dxyFromBS>0.2 && zGoldenDau2dxyFromBS>0.2" ) ;
  zMassBothDaudxyFromBSHigher0_2->Write();
  delete zMassBothDaudxyFromBSHigher0_2;



  dir->cd();
  TH1F * zMassOneDaudxyFromBSHigher0_2  = getTH1Histo(Events, "zMassOneDaudxyFromBSHigher0_2", "zGoldenMass", 200, -1, 1,  cut_zGolden + "zGoldenDau1dxyFromBS>0.2 || zGoldenDau2dxyFromBS>0.2" ) ;
  zMassOneDaudxyFromBSHigher0_2->Write();
  delete zMassOneDaudxyFromBSHigher0_2;
 

  // isTrackerMuon
  dir->cd(); 
  TH1F * zMassBothDauNoTrackerMuon    = getTH1Histo(Events, "zMassBothDauNoTrackerMuon", "zGoldenMass", 200, 0, 200,  cut_zGolden + "zGoldenDau1TrackerMuonBit==0 && zGoldenDau2TrackerMuonBit==0");
  zMassBothDauNoTrackerMuon->Write();
  delete zMassBothDauNoTrackerMuon;

  dir->cd(); 
  TH1F * zMassOneDauNoTrackerMuon    = getTH1Histo(Events, "zMassOneDauNoTrackerMuon", "zGoldenMass", 200, 0, 200,  cut_zGolden + "zGoldenDau1TrackerMuonBit==0 || zGoldenDau2TrackerMuonBit==0");
  zMassOneDauNoTrackerMuon->Write();
  delete zMassOneDauNoTrackerMuon;
  


  // Eta distribution if MuonHits is zero
  
  dir->cd();
  TH1F * dauEtaNofMuonHits0 = getTH1Histo(Events, "dauEtaNofMuonHits0", "zGoldenDau1Eta", 200, -5, 5,  cut_zGolden +  "zGoldenDau1NofMuonHits==0" );
  TH1F * dau2EtaNofMuonHits0 = getTH1Histo(Events, "dau2EtaNofMuonHits0", "zGoldenDau2Eta", 200, -5, 5,  cut_zGolden +  "zGoldenDau2NofMuonHits==0" );
  dauEtaNofMuonHits0->Add(dau2EtaNofMuonHits0);
  dauEtaNofMuonHits0->Write();
  delete dauEtaNofMuonHits0;
  delete dau2EtaNofMuonHits0;
  

  // Correlation study

  // *** Chi2 vs MuonHits  ***
 
  dir->cd();
  TH2F * Chi2VsMuonHits = new TH2F("Chi2VsMuonHits", "Chi2VsMuonHits", 100, 0, 60, 100, 0, 6);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events->Project("Chi2VsMuonHits", "zGoldenDau1Chi2:zGoldenDau1NofMuonHits", cut_zGolden);         
  Chi2VsMuonHits->SetDrawOption("Box"); 

  Chi2VsMuonHits->Write();

  delete Chi2VsMuonHits;


  // *** Chi2 vs StripHits  ***
 
  dir->cd();
  TH2F * Chi2VsStripHits = new TH2F("Chi2VsStripHits", "Chi2VsStripHits", 100, 0, 30, 100, 0, 6);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events->Project("Chi2VsStripHits", "zGoldenDau1Chi2:zGoldenDau1NofStripHits", cut_zGolden);         
  Chi2VsStripHits->SetDrawOption("Box"); 

  Chi2VsStripHits->Write();

  delete Chi2VsStripHits;

   // *** MuonHits vs Eta  ***
 
  dir->cd();
  TH2F * MuonHitsVsEta = new TH2F("MuonHitsVsEta", "MuonHitsVsEta", 100, -2.5, 2.5, 100, 0, 60);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events->Project("MuonHitsVsEta", "zGoldenDau1NofMuonHits:zGoldenDau1Eta", cut_zGolden);         
  MuonHitsVsEta->SetDrawOption("Box"); 

  MuonHitsVsEta->Write();

  delete MuonHitsVsEta;


  output_file->cd("/");

  
 

  //   output_file->Close(); 
 
}





void qualityStudiesZGlbTrk(TFile *output_file, TChain * Events){

  gStyle->SetOptStat();
  gROOT->SetStyle("Plain");
  using namespace std;
  



  // zMuTrk plots
  TCut cut_zMuTrk("zMuTrkMass>60 && zMuTrkMass<120 && zMuTrkDau1Pt> 20 && zMuTrkDau2Pt>20 && zMuTrkDau1TrkIso< 3.0 && zMuTrkDau2TrkIso < 3.0 && zMuTrkDau1Eta<2.1 &&  zMuTrkDau2Eta<2.1");
  TDirectory * dir = output_file->mkdir("goodZToMuMuOneTrackPlots");
  dir->cd();



  TH1F * zMass = getTH1Histo(Events, "zMass", "zMuTrkMass", 200, 0, 200,  cut_zMuTrk) ;
  zMass->Write();
  delete zMass;

  //Quality checks
  // Chi2

  // quality variables
  // caveat: I'm  requiring isolations
  TH1F * dau1Chi2 = getTH1Histo(Events, "dau1Chi2", "zMuTrkDau1Chi2", 1000, 0, 100,  cut_zMuTrk) ;
  TH1F * dau2Chi2 = getTH1Histo(Events, "dau2Chi2", "zMuTrkDau2Chi2", 1000, 0, 100,  cut_zMuTrk) ;
 
  dau1Chi2->Write();
  dau2Chi2->Write();
  delete dau1Chi2;
  delete dau2Chi2;
   

  dir->cd();
  TH1F * zMassDau1Chi2Higher10 = getTH1Histo(Events, "zMassDau1Chi2Higher10", "zMuTrkMass", 200, 0, 200,  cut_zMuTrk +"zMuTrkDau1Chi2>10") ;  
  zMassDau1Chi2Higher10->Write();
  delete   zMassDau1Chi2Higher10;

  dir->cd();
  TH1F * zMassOneDauChi2Higher10 = getTH1Histo(Events, "zMassOneDauChi2Higher10", "zMuTrkMass", 200, 0, 200,  cut_zMuTrk +"zMuTrkDau1Chi2>10 || zMuTrkDau2Chi2>10") ;  
  zMassOneDauChi2Higher10->Write();
  delete   zMassOneDauChi2Higher10;

  dir->cd();
  TH1F * zMassBothDauChi2Higher10 = getTH1Histo(Events, "zMassBothDauChi2Higher10", "zMuTrkMass", 200, 0, 200,  cut_zMuTrk +"zMuTrkDau1Chi2>10 && zMuTrkDau2Chi2>10") ;
  zMassBothDauChi2Higher10->Write();
  delete   zMassBothDauChi2Higher10;


  dir->cd();  
  TH1F *dau1Chi2NofMuonHits0  = getTH1Histo(Events, "dau1Chi2NofMuonHits0", "zMuTrkDau1Chi2", 1000, 0, 100,  cut_zMuTrk + "zMuTrkDau1NofMuonHits==0") ;
  dau1Chi2NofMuonHits0->Write();
  delete dau1Chi2NofMuonHits0;



  dir->cd();
  TH1F *dau1SaChi2NofSaMuonHits0  = getTH1Histo(Events, "dau1SaChi2NofSaMuonHits0", "zMuTrkDau1SaChi2", 1000, 0, 100,  cut_zMuTrk + "zMuTrkDau1SaNofMuonHits==0") ;
  dau1SaChi2NofSaMuonHits0->Write();
  delete dau1SaChi2NofSaMuonHits0;



  TH1F *dau1Chi2NofStripHits0  = getTH1Histo(Events, "dau1Chi2NofStripHits0", "zMuTrkDau1Chi2", 1000, 0, 100,  cut_zMuTrk + "zMuTrkDau1TrkNofStripHits<10") ;
  dau1Chi2NofStripHits0->Write();
  delete dau1Chi2NofStripHits0;

  

  
 

  

// Number of  Strips Hits for inner track
  dir->cd();
  // caveat: I'm  requiring isolations
   
  TH1F *dau1TrkNofStripHits  = getTH1Histo(Events, "dau1TrkNofStripHits", "zMuTrkDau1TrkNofStripHits", 100, 0, 100,  cut_zMuTrk ) ;
  TH1F *dau2TrkNofStripHits  = getTH1Histo(Events, "dau2TrkNofStripHits", "zMuTrkDau2TrkNofStripHits", 100, 0, 100,  cut_zMuTrk ) ;
  dau1TrkNofStripHits->Write();
  dau2TrkNofStripHits->Write();
  delete dau2TrkNofStripHits;
  delete dau2TrkNofStripHits;


dir->cd();
  TH1F * zMassDau1TrkNofStripsHitsLower10 = getTH1Histo(Events, "zMassDau1TrkNofStripsHitsLower10","zMuTrkMass", 200, 0, 200,cut_zMuTrk +"zMuTrkDau1TrkNofStripHits<10" );
  zMassDau1TrkNofStripsHitsLower10->Write();
  delete zMassDau1TrkNofStripsHitsLower10;

  dir->cd();
  TH1F * zMassBothDauTrkNofStripsHitsLower10 = getTH1Histo(Events, "zMassBothDauTrkNofStripsHitsLower10","zMuTrkMass", 200, 0, 200,cut_zMuTrk +"zMuTrkDau1TrkNofStripHits<10 && zMuTrkDau2TrkNofStripHits<10" );
  zMassBothDauTrkNofStripsHitsLower10->Write();
  delete zMassBothDauTrkNofStripsHitsLower10;

  dir->cd();
  TH1F * zMassOneDauTrkNofStripsHitsLower10 = getTH1Histo(Events, "zMassOneDauTrkNofStripsHitsLower10","zMuTrkMass", 200, 0, 200,cut_zMuTrk +"zMuTrkDau1TrkNofStripHits<10 || zMuTrkDau2TrkNofStripHits<10" );
  zMassOneDauTrkNofStripsHitsLower10->Write();
  delete zMassOneDauTrkNofStripsHitsLower10;

  

  


 // Number of  Pixel Hits for inner track
 dir->cd();
  // caveat: I'm  requiring isolations
   
  TH1F *dau1TrkNofPixelHits  = getTH1Histo(Events, "dau1TrkNofPixelHits", "zMuTrkDau1TrkNofPixelHits", 100, 0, 100,  cut_zMuTrk ) ;
  TH1F *dau2TrkNofPixelHits  = getTH1Histo(Events, "dau2TrkNofPixelHits", "zMuTrkDau1TrkNofPixelHits", 100, 0, 100,  cut_zMuTrk ) ;
  dau1TrkNofPixelHits->Write();
  dau2TrkNofPixelHits->Write();
  delete dau1TrkNofPixelHits;
  delete dau2TrkNofPixelHits;



  dir->cd();
  TH1F * zMassDau1TrkNofPixelsHitsLower1 = getTH1Histo(Events, "zMassDau1TrkNofPixelsHitsLower1","zMuTrkMass", 200, 0, 200,cut_zMuTrk +"zMuTrkDau1TrkNofPixelHits<1" );
  zMassDau1TrkNofPixelsHitsLower1->Write();
  delete zMassDau1TrkNofPixelsHitsLower1;

  dir->cd();
  TH1F * zMassBothDauTrkNofPixelsHitsLower1 = getTH1Histo(Events, "zMassBothDauTrkNofPixelsHitsLower1","zMuTrkMass", 200, 0, 200,cut_zMuTrk +"zMuTrkDau1TrkNofPixelHits<1 && zMuTrkDau2TrkNofPixelHits<1" );
  zMassBothDauTrkNofPixelsHitsLower1->Write();
  delete zMassBothDauTrkNofPixelsHitsLower1;

  dir->cd();
  TH1F * zMassOneDauTrkNofPixelsHitsLower1 = getTH1Histo(Events, "zMassOneDauTrkNofPixelsHitsLower1","zMuTrkMass", 200, 0, 200,cut_zMuTrk +"zMuTrkDau1TrkNofPixelHits<1 || zMuTrkDau2TrkNofPixelHits<1" );
  zMassOneDauTrkNofPixelsHitsLower1->Write();
  delete zMassOneDauTrkNofPixelsHitsLower1;
  





  // Number of  Muon Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dau1NofMuonHits  = getTH1Histo(Events, "dau1NofMuonHits", "zMuTrkDau1NofMuonHits", 100, 0, 100,  cut_zMuTrk ) ;
  dau1NofMuonHits->Write();
  delete dau1NofMuonHits;


  dir->cd();
  TH1F * zMassDau1NofMuonsHitsLower1 = getTH1Histo(Events, "zMassDau1NofMuonsHitsLower1","zMuTrkMass", 200, 0, 200,cut_zMuTrk +"zMuTrkDau1NofMuonHits<1 " );
  zMassDau1NofMuonsHitsLower1->Write();
  delete zMassDau1NofMuonsHitsLower1;
  

  // Number of  Muon Hits for outer track
 dir->cd();
  // caveat: I'm  requiring isolations
  
  TH1F *dau1SaNofMuonHits  = getTH1Histo(Events, "dau1SaNofMuonHits", "zMuTrkDau1SaNofMuonHits", 100, 0, 100,  cut_zMuTrk ) ;
  dau1SaNofMuonHits->Write();
  delete dau1SaNofMuonHits;



  dir->cd();
  TH1F * zMassDau1SaNofMuonsHitsLower1 = getTH1Histo(Events, "zMassDau1SaNofMuonsHitsLower1","zMuTrkMass", 200, 0, 200,cut_zMuTrk +"zMuTrkDau1SaNofMuonHits<1 " );
  zMassDau1SaNofMuonsHitsLower1->Write();
  delete zMassDau1SaNofMuonsHitsLower1;



  // dxyFromBS
  dir->cd();

  TH1F *dauDxy  = getTH1Histo(Events, "dauDxy", "zMuTrkDau1dxyFromBS", 200, -1, 1,  cut_zMuTrk ) ;
  TH1F *dau2Dxy  = getTH1Histo(Events, "dau2Dxy", "zMuTrkDau2dxyFromBS", 200, -1, 1,  cut_zMuTrk ) ;
  dauDxy->Add(dau2Dxy);
  dauDxy->Write();
  delete dauDxy;
  delete dau2Dxy;


  TH1F * zMassBothDaudxyFromBSHigher0_2  = getTH1Histo(Events, "zMassBothDaudxyFromBSHigher0_2", "zMuTrkMass", 200, -1, 1,  cut_zMuTrk + "zMuTrkDau1dxyFromBS>0.2 && zMuTrkDau2dxyFromBS>0.2" ) ;
  zMassBothDaudxyFromBSHigher0_2->Write();
  delete zMassBothDaudxyFromBSHigher0_2;



  dir->cd();
  TH1F * zMassOneDaudxyFromBSHigher0_2  = getTH1Histo(Events, "zMassOneDaudxyFromBSHigher0_2", "zMuTrkMass", 200, -1, 1,  cut_zMuTrk + "zMuTrkDau1dxyFromBS>0.2 || zMuTrkDau2dxyFromBS>0.2" ) ;
  zMassOneDaudxyFromBSHigher0_2->Write();
  delete zMassOneDaudxyFromBSHigher0_2;
 

  // isTrackerMuon
  dir->cd(); 


  TH1F * zMassDau1NoTrackerMuon    = getTH1Histo(Events, "zMassDau1NoTrackerMuon", "zMuTrkMass", 200, 0, 200,  cut_zMuTrk + "zMuTrkDau1TrackerMuonBit==0 ");
  zMassDau1NoTrackerMuon->Write();
  delete zMassDau1NoTrackerMuon;
  


  // Eta distribution if MuonHits is zero
  
  dir->cd();
  TH1F * dau1EtaNofMuonHits0 = getTH1Histo(Events, "dau1EtaNofMuonHits0", "zMuTrkDau1Eta", 200, -5, 5,  cut_zMuTrk +  "zMuTrkDau1NofMuonHits==0" );
  dau1EtaNofMuonHits0->Write();
  delete dau1EtaNofMuonHits0;

  

 
}








void qualityStudies(){

TChain * Events= new TChain("Events"); 
  
  int nFiles = 220;

  for(int j=1;j<nFiles;++j){
    ostringstream oss;
    oss<<j;
    //string name= "zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_"+oss.str()+"_None.root";
    string name= "/tmp/degrutto/NtupleLooseTestNew_oneshot_all_Test_"+oss.str()+"_None.root";
    Events->Add(name.c_str());
  }

TFile * output_file = TFile::Open("histo_qcd.root", "RECREATE");
qualityStudiesZGolden(output_file, Events);
 qualityStudiesZGoldenNotIso(output_file, Events);
  qualityStudiesZGlbTrk(output_file, Events);
 qualityStudiesZMuSta(output_file, Events);
   output_file->Close(); 
}
