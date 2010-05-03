#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TTree.h"
#include "TAxis.h"
#include "TCanvas.h"
//#if !defined(__CINT__) && !defined(__MAKECINT__)                                                                                                          
#include <string>
#include <iostream>
//#endif   

void setGraphics(TH1F *histo){

  histo->SetFillColor(kAzure+7);
  //histo->SetLineWidth(2);
  histo->SetLineColor(kBlue+1);
}




void qualityStudies(){

  gStyle->SetOptStat();
  gROOT->SetStyle("Plain");
  using namespace std;

  //  #include <exception>;

  //    TFile *file = TFile::Open("rfio:/castor/cern.ch/user/d/degrutto/incl15WithBsPv/NtupleLoose_test_inclu15_1_2.root");
  // TFile *file = TFile::Open("../NutpleLooseTestNew_oneshot_all_10_1.root");
//TFile *file = TFile::Open("rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/NtupleLoose_test.root");
//    TTree * Events = dynamic_cast< TTree *> (file->Get("Events"));

  TChain Events("Events"); 
  
  Events.Add("../NutpleLooseTestNew_oneshot_all_10_1.root");
  Events.Add("../NutpleLooseTestNew_oneshot_all_11_1.root");
  Events.Add("../NutpleLooseTestNew_oneshot_all_12_1.root");
  Events.Add("../NutpleLooseTestNew_oneshot_all_13_1.root");
  Events.Add("../NutpleLooseTestNew_oneshot_all_4_1.root");
  Events.Add("../NutpleLooseTestNew_oneshot_all_5_1.root");
  Events.Add("../NutpleLooseTestNew_oneshot_all_6_1.root");
  Events.Add("../NutpleLooseTestNew_oneshot_all_7_1.root");
  Events.Add("../NutpleLooseTestNew_oneshot_all_8_1.root");
  
  TFile * output_file = TFile::Open("histo.root", "RECREATE");
  // TFile * output_file = TFile::Open("histo_test.root", "RECREATE");
  
  // zGolden plots
  TCut cut_zGolden("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1Iso< 3.0 && zGoldenDau2Iso < 3.0 && zGoldenDau1Eta<2.1 &&  zGoldenDau2Eta<2.1");
  TDirectory * dir = output_file->mkdir("goodZToMuMuPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  //  Events.Draw("zGoldenMass");
  Events.Project("zMass", "zGoldenMass", cut_zGolden );
  cout<<"Number of zGolden : "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  //Quality checks


 // Chi2
  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauChi2Higher10", "zMassOneDauChi2Higher10", 200, 0, 200);
  //  Events.Draw("zGoldenMass");
  Events.Project("zMassOneDauChi2Higher10", "zGoldenMass", cut_zGolden +"zGoldenDau1Chi2>10 || zGoldenDau2Chi2>10");
  setGraphics(zMass);
  zMass->Write();
  cout<<"Number of zCandidate with at least one daughter with Chi2 higher than 10: "<<zMass->GetEntries()<<endl;
  delete zMass;

  dir->cd();
  TH1F * zMass = new TH1F("zMassBothDauChi2Higher10","zMassBothDauChi2Higher10" , 200, 0, 200);
  //  Events.Draw("zGoldenMass");
  Events.Project("zMassBothDauChi2Higher10", "zGoldenMass", cut_zGolden +"zGoldenDau1Chi2>10 && zGoldenDau2Chi2>10");
  cout<<"Number of zCandidate with both daughters with Chi2 higher than 10: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  dir->cd();
  TH1F * zMass = new TH1F("zChi2NofMuonHits0","zChi2NofMuonHits0" , 200, 0, 200);
  //  Events.Draw("zGoldenMass");
  Events.Project("zChi2NofMuonHits0", "zGoldenDau1Chi2", cut_zGolden +  "zGoldenDau1NofMuonHits==0");
  Events.Project("zChi2NofMuonHits0", "zGoldenDau2Chi2", cut_zGolden +  "zGoldenDau2NofMuonHits==0");
  setGraphics(zMass);
  zMass->Write();
  //cout<<"Number of zCandidate with at least one daughter with Chi2 higher: "<<zMass->GetEntries()<<endl;
  delete zMass;


  dir->cd();
  TH1F * zMass = new TH1F("zChi2NofStripsHits0", "zChi2NofStripsHits0", 200, 0, 20);
  //  Events.Draw("zGoldenMass");
  Events.Project("zChi2NofStripsHits0", "zGoldenDau1Chi2", cut_zGolden +  "zGoldenDau1NofStripHits<10");
  Events.Project("zChi2NofStripsHits0", "zGoldenDau2Chi2", cut_zGolden +  "zGoldenDau2NofStripHits<10");
  setGraphics(zMass);
  zMass->Write();
  //cout<<"Number of zCandidate with at least one daughter with Chi2 higher: "<<zMass->GetEntries()<<endl;
  delete zMass;


  dir->cd();
  TH1F * zMass = new TH1F("zSaChi2NofMuonHits0","zSaChi2NofMuonHits0" , 200, 0, 200);
  //  Events.Draw("zGoldenMass");
  Events.Project("zSaChi2NofMuonHits0", "zGoldenDau1SaChi2", cut_zGolden +  "zGoldenDau1NofMuonHits==0");
  Events.Project("zSaChi2NofMuonHits0", "zGoldenDau2SaChi2", cut_zGolden +  "zGoldenDau2NofMuonHits==0");
  setGraphics(zMass);
  zMass->Write();
  //cout<<"Number of zCandidate with at least one daughter with Chi2 higher: "<<zMass->GetEntries()<<endl;
  delete zMass;

  
 

 // Number of  Strips Hits
  dir->cd();
  TH1F * zMass = new TH1F("zMassBothDauNofStripsHitsHigher10","zMassBothDauNofStripsHitsHigher10" , 200, 0, 200);
  Events.Project("zMassBothDauNofStripsHitsHigher10", "zGoldenMass", cut_zGolden +"zGoldenDau1NofStripHits<10 && zGoldenDau2NofStripHits<10");
  cout<<"Number of zCandidate with both daughters with nmber of strips hits lower than 10: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauNofStripsHitsHigher10","zMassOneDauNofStripsHitsHigher10" , 200, 0, 200);
  Events.Project("zMassOneDauNofStripsHitsHigher10", "zGoldenMass", cut_zGolden + "zGoldenDau1NofStripHits<10 || zGoldenDau2NofStripHits<10"  );
  cout<<"Number of zCandidate with at least one daughter with number of strips hits lower than 10: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;
  

 // Number of  Pixel Hits
  dir->cd();  
  TH1F * zMass = new TH1F("zMassBothDauNofPixelHitsHigher10","zMassBothDauNofPixelHitsHigher10" , 200, 0, 200);
  Events.Project("zMassBothDauNofPixelHitsHigher10", "zGoldenMass", cut_zGolden + "zGoldenDau1NofPixelHits==0 && (zGoldenDau2NofPixelHits==0 &&  zGoldenDau1Chi2<10 || zGoldenDau2Chi2<10)"  );
  cout<<"Number of zCandidate with both daughters with number of pixel hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;



 // Number of  Pixel Hits
  dir->cd();  
  TH1F * zMass = new TH1F("zMassBothDauNofPixelHitsHigher10","zMassBothDauNofPixelHitsHigher10" , 200, 0, 200);
  Events.Project("zMassBothDauNofPixelHitsHigher10", "zGoldenMass", cut_zGolden + "zGoldenDau1NofPixelHits==0 && (zGoldenDau2NofPixelHits==0 &&  zGoldenDau1Chi2<10 || zGoldenDau2Chi2<10)"  );
  cout<<"Number of zCandidate with both daughters with number of pixel hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;


  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauNofPixelHitsHigher10","zMassOneDauNofPixelHitsHigher10" , 200, 0, 200);
  Events.Project("zMassOneDauNofPixelHitsHigher10", "zGoldenMass", cut_zGolden + "zGoldenDau1NofPixelHits==0 || zGoldenDau2NofPixelHits==0"  );
  cout<<"Number of zCandidate with at least one daughter with number of pixel hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;
  

  // Number of  Muon Hits
  dir->cd();  
  TH1F * zMass = new TH1F("zMassBothDauNofMuonHitsLower10", "zMass", 200, 0, 200);
  Events.Project("zMassBothDauNofMuonHitsLower10", "zGoldenMass", cut_zGolden + "zGoldenDau1NofMuonHits==0 && zGoldenDau2NofMuonHits==0"  );
  cout<<"Number of zCandidate with both daughters with number of muon hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauNofMuonHitsLower0","zMassOneDauNofMuonHitsLower0" , 200, 0, 200);
  Events.Project("zMassOneDauNofMuonHitsLower0", "zGoldenMass", cut_zGolden + "zGoldenDau1NofMuonHits==0 || zGoldenDau2NofMuonHits==0"  );
  cout<<"Number of zCandidate with at least one daughter with number of muon hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;
 

  // dxyFromBS
  dir->cd();
  TH1F * zMass = new TH1F("zMassBothDaudxyFromBSHigher0_02","zMassBothDaudxyFromBSHigher0_02" , 200, 0, 200);
  Events.Project("zMassBothDaudxyFromBSHigher0_02", "zGoldenMass", cut_zGolden + "zGoldenDau1dxyFromBS>0.02 && zGoldenDau2dxyFromBS>0.02"  );
  cout<<"Number of zCandidate with both daughters with dxyFromBS higher than 0.02: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDaudxyFromBSHigher0_02", "zMassOneDaudxyFromBSHigher0_02", 200, 0, 200);
  Events.Project("zMassOneDaudxyFromBSHigher0_02", "zGoldenMass", cut_zGolden + "zGoldenDau1dxyFromBS>0.02 || zGoldenDau2dxyFromBS>0.02"  );
  cout<<"Number of zCandidate with at least one daughter with dxyFromBS higher than 0.02: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;
 

  // isTrackerMuon
  dir->cd(); 
  TH1F * zMass = new TH1F("zMassBothDauNoTrackerMuon","zMassBothDauNoTrackerMuon" , 200, 0, 200);
  Events.Project("zMassBothDauNoTrackerMuon", "zGoldenMass", cut_zGolden + "zGoldenDau1TrackerMuonBit==0 && zGoldenDau2TrackerMuonBit==0"  );
  cout<<"Number of zCandidate with both daughters that are not TrackerMuons: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;


  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauNoTrackerMuon", "zMassOneDauNoTrackerMuon", 200, 0, 200);
  Events.Project("zMassOneDauNoTrackerMuon", "zGoldenMass", cut_zGolden + "zGoldenDau1TrackerMuonBit==0 || zGoldenDau2TrackerMuonBit==0"  );
  cout<<"Number of zCandidate with at least one daughter not TrackerMuon : "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;
  


  // Eta distribution if MuonHits is zero
  
  dir->cd();
  TH1F * zEta = new TH1F("zEtaBothDauNofMuonHitsLower10","zEtaBothDauNofMuonHitsLower10" , 200, -3, 3);
  Events.Project("zEtaBothDauNofMuonHitsLower10", "zGoldenEta", cut_zGolden + "zGoldenDau1NofMuonHits==0 && zGoldenDau2NofMuonHits==0"  );
  setGraphics(zEta);
  zEta->Write();
  delete zEta;

  dir->cd();
  TH1F * zEta = new TH1F("zEtaOneDauNofMuonHitsLower0","zEtaOneDauNofMuonHitsLower0" , 200, -3, 3);
  Events.Project("zEtaOneDauNofMuonHitsLower0", "zGoldenEta", cut_zGolden + "zGoldenDau1NofMuonHits==0 || zGoldenDau2NofMuonHits==0"  );
  setGraphics(zEta);
  zEta->Write();
  delete zEta;
  

  // Correlation study

  // *** Chi2 vs MuonHits  ***
 
  dir->cd();
  TH2F * Chi2VsMuonHits = new TH2F("Chi2VsMuonHits", "Chi2VsMuonHits", 100, 0, 60, 100, 0, 6);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events.Project("Chi2VsMuonHits", "zGoldenDau1Chi2:zGoldenDau1NofMuonHits", cut_zGolden);         
  Chi2VsMuonHits->SetDrawOption("Box"); 

  Chi2VsMuonHits->Write();

  delete Chi2VsMuonHits;


  // *** Chi2 vs StripHits  ***
 
  dir->cd();
  TH2F * Chi2VsStripHits = new TH2F("Chi2VsStripHits", "Chi2VsStripHits", 100, 0, 30, 100, 0, 6);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events.Project("Chi2VsStripHits", "zGoldenDau1Chi2:zGoldenDau1NofStripHits", cut_zGolden);         
  Chi2VsStripHits->SetDrawOption("Box"); 

  Chi2VsStripHits->Write();

  delete Chi2VsStripHits;

   // *** MuonHits vs Eta  ***
 
  dir->cd();
  TH2F * MuonHitsVsEta = new TH2F("MuonHitsVsEta", "MuonHitsVsEta", 100, -2.5, 2.5, 100, 0, 60);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events.Project("MuonHitsVsEta", "zGoldenDau1NofMuonHits:zGoldenDau1Eta", cut_zGolden);         
  MuonHitsVsEta->SetDrawOption("Box"); 

  MuonHitsVsEta->Write();

  delete MuonHitsVsEta;


  output_file->cd("/");

  
 

   output_file->Close(); 
 
}
