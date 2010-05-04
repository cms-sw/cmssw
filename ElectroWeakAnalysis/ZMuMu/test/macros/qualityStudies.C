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




void qualityStudiesZGolden(TFile * output_file){

  gStyle->SetOptStat();
  gROOT->SetStyle("Plain");
  using namespace std;

  //  #include <exception>;

  //    TFile *file = TFile::Open("rfio:/castor/cern.ch/user/d/degrutto/incl15WithBsPv/NtupleLoose_test_inclu15_1_2.root");
  // TFile *file = TFile::Open("../NutpleLooseTestNew_oneshot_all_10_1.root");
//TFile *file = TFile::Open("rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/NtupleLoose_test.root");
//    TTree * Events = dynamic_cast< TTree *> (file->Get("Events"));

  TChain Events("Events"); 
  
  Events.Add("NtupleLooseTestNew_oneshot_all_Test_1_None.root");
 Events.Add("NtupleLooseTestNew_oneshot_all_Test_2_None.root");
 Events.Add("NtupleLooseTestNew_oneshot_all_Test_6_None.root");
  //  Events.Add("../NutpleLooseTestNew_oneshot_all_11_1.root");
  //  Events.Add("../NutpleLooseTestNew_oneshot_all_12_1.root");
  //Events.Add("../NutpleLooseTestNew_oneshot_all_13_1.root");
  //Events.Add("../NutpleLooseTestNew_oneshot_all_4_1.root");
  //Events.Add("../NutpleLooseTestNew_oneshot_all_5_1.root");
  //Events.Add("../NutpleLooseTestNew_oneshot_all_6_1.root");
  //Events.Add("../NutpleLooseTestNew_oneshot_all_7_1.root");
  //Events.Add("../NutpleLooseTestNew_oneshot_all_8_1.root");
  
 //   TFile * output_file  = TFile::Open("histo_test.root", "RECREATE");
  
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

 // quality variables
  // caveat: I'm  requiring isolations
  TH1F * dauChi2 = new TH1F("duaChi2", "dauChi2", 1000, 0, 100);
  TH1F * h2 = new TH1F("h2", "h2", 1000, 0, 100);
  Events.Project("dua1Chi2", "zGoldenDau1Chi2", cut_zGolden );
  Events.Project("h2", "zGoldenDau2Chi2", cut_zGolden );
  dauChi2->Add(h2);
  dauChi2->Write();
  delete dauChi2;
  delete h2;
   

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

  
 

 // Number of  Strips Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  TH1F * dauNofStripHits = new TH1F("dauNofStripHits", "dauNofStripHits", 100, 0, 100);
  TH1F * h2 = new TH1F("h2", "h2", 100, 0, 100);
  Events.Project("dauNofStripHits", "zGoldenDau1NofStripHits", cut_zGolden );
  Events.Project("h2", "zGoldenDau2NofStripHits", cut_zGolden );
  dauNofStripHits->Add(h2);
  dauNofStripHits->Write();
  delete dauNofStripHits;
  delete h2;
 

  dir->cd();
  TH1F * zMass = new TH1F("zMassBothDauNofStripsHitsLower10","zMassBothDauNofStripsHitsLower10" , 200, 0, 200);
  Events.Project("zMassBothDauNofStripsHitsLower10", "zGoldenMass", cut_zGolden +"zGoldenDau1NofStripHits<10 && zGoldenDau2NofStripHits<10");
  cout<<"Number of zCandidate with both daughters with nmber of strips hits lower than 10: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauNofStripsHitsLower10","zMassOneDauNofStripsHitsLower10" , 200, 0, 200);
  Events.Project("zMassOneDauNofStripsHitsLower10", "zGoldenMass", cut_zGolden + "zGoldenDau1NofStripHits<10 || zGoldenDau2NofStripHits<10"  );
  cout<<"Number of zCandidate with at least one daughter with number of strips hits lower than 10: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;
  

// Number of  Strips Hits for inner track
  dir->cd();
  // caveat: I'm  requiring isolations
  TH1F * dauTrkNofStripHits = new TH1F("dauTrkNofStripHits", "dauTrkNofStripHits", 100, 0, 100);
  TH1F * h2 = new TH1F("h2", "h2", 100, 0, 100);
  Events.Project("dauTrkNofStripHits", "zGoldenDau1TrkNofStripHits", cut_zGolden );
  Events.Project("h2", "zGoldenDau2TrkNofStripHits", cut_zGolden );
  dauTrkNofStripHits->Add(h2);
  dauTrkNofStripHits->Write();
  delete dauTrkNofStripHits;
  delete h2;
 

  dir->cd();
  TH1F * zMass = new TH1F("zMassBothDauTrkNofStripsHitsLower10","zMassBothDauTrkNofStripsHitsLower10" , 200, 0, 200);
  Events.Project("zMassBothDauTrkNofStripsHitsLower10", "zGoldenMass", cut_zGolden +"zGoldenDau1TrkNofStripHits<10 && zGoldenDau2TrkNofStripHits<10");
  cout<<"Number of zCandidate with both daughters->innerTrack() with nmber of strips hits lower than 10: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauTrkNofStripsHitsLower10","zMassOneDauTrkNofStripsHitsLower10" , 200, 0, 200);
  Events.Project("zMassOneDauTrkNofStripsHitsLower10", "zGoldenMass", cut_zGolden + "zGoldenDau1TrkNofStripHits<10 || zGoldenDau2TrkNofStripHits<10"  );
  cout<<"Number of zCandidate with at least one daughter->innerTrack with number of strips hits lower than 10: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;
  





 // Number of  Pixel Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  TH1F * dauNofPixelHits = new TH1F("dauNofPixelHits", "dauNofPixelHits", 100, 0, 100);
  TH1F * h2 = new TH1F("h2", "h2", 100, 0, 100);
  Events.Project("dauNofPixelHits", "zGoldenDau1NofPixelHits", cut_zGolden );
  Events.Project("h2", "zGoldenDau2NofPixelHits", cut_zGolden );
  dauNofPixelHits->Add(h2);
  dauNofPixelHits->Write();
  delete dauNofPixelHits;
  delete h2;  
  
 // Number of  Pixel Hits
  dir->cd();  
  TH1F * zMass = new TH1F("zMassBothDauNofPixelHitsLower1","zMassBothDauNofPixelHitsLower1" , 200, 0, 200);
  //  Events.Project("zMassBothDauNofPixelHitsLower1", "zGoldenMass", cut_zGolden + "zGoldenDau1NofPixelHits==0 && (zGoldenDau2NofPixelHits==0 &&  zGoldenDau1Chi2<10 || zGoldenDau2Chi2<10)"  );
  Events.Project("zMassBothDauNofPixelHitsLower1", "zGoldenMass", cut_zGolden + "zGoldenDau1NofPixelHits==0 && zGoldenDau2NofPixelHits==0"  );
  cout<<"Number of zCandidate with both daughters with number of pixel hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;


  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauNofPixelHitsLower1","zMassOneDauNofPixelHitsLower1" , 200, 0, 200);
  Events.Project("zMassOneDauNofPixelHitsLower1", "zGoldenMass", cut_zGolden + "zGoldenDau1NofPixelHits==0 || zGoldenDau2NofPixelHits==0"  );
  cout<<"Number of zCandidate with at least one daughter with number of pixel hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;


 // Number of  Pixel Hits for inner track
  dir->cd();
  // caveat: I'm  requiring isolations
  TH1F * dauTrkNofPixelHits = new TH1F("dauTrkNofPixelHits", "dauTrkNofPixelHits", 100, 0, 100);
  TH1F * h2 = new TH1F("h2", "h2", 100, 0, 100);
  Events.Project("dauTrkNofPixelHits", "zGoldenDau1TrkNofPixelHits", cut_zGolden );
  Events.Project("h2", "zGoldenDau2TrkNofPixelHits", cut_zGolden );
  dauTrkNofPixelHits->Add(h2);
  dauTrkNofPixelHits->Write();
  delete dauTrkNofPixelHits;
  delete h2;  
  
 // Number of  Pixel Hits
  dir->cd();  
  TH1F * zMass = new TH1F("zMassBothDauTrkNofPixelHitsLower1","zMassBothDauTrkNofPixelHitsLower1" , 200, 0, 200);
  //  Events.Project("zMassBothDauTrkNofPixelHitsLower1", "zGoldenMass", cut_zGolden + "zGoldenDauTrk1NofPixelHits==0 && (zGoldenDauTrk2NofPixelHits==0 &&  zGoldenDauTrk1Chi2<10 || zGoldenDauTrk2Chi2<10)"  );
  Events.Project("zMassBothDauTrkNofPixelHitsLower1", "zGoldenMass", cut_zGolden + "zGoldenDau1TrkNofPixelHits==0 && zGoldenDau2TrkNofPixelHits==0"  );
  cout<<"Number of zCandidate with both daughters->innerTrack() with number of pixel hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;


  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauTrkNofPixelHitsLower1","zMassOneDauTrkNofPixelHitsLower1" , 200, 0, 200);
  Events.Project("zMassOneDauTrkNofPixelHitsLower1", "zGoldenMass", cut_zGolden + "zGoldenDau1TrkNofPixelHits==0 || zGoldenDau2TrkNofPixelHits==0"  );
  cout<<"Number of zCandidate with at least one daughter->innerTrack() with number of pixel hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;
  






  // Number of  Muon Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  TH1F * dauNofMuonHits = new TH1F("dauNofMuonHits", "dauNofMuonHits", 100, 0, 100);
  TH1F * h2 = new TH1F("h2", "h2", 100, 0, 100);
  Events.Project("dauNofMuonHits", "zGoldenDau1NofMuonHits", cut_zGolden );
  Events.Project("h2", "zGoldenDau2NofMuonHits", cut_zGolden );
  dauNofMuonHits->Add(h2);
  dauNofMuonHits->Write();
  delete dauNofMuonHits;
  delete h2;



  dir->cd();  
  TH1F * zMass = new TH1F("zMassBothDauNofMuonHitsLower1", "zMass", 200, 0, 200);
  Events.Project("zMassBothDauNofMuonHitsLower1", "zGoldenMass", cut_zGolden + "zGoldenDau1NofMuonHits==0 && zGoldenDau2NofMuonHits==0"  );
  cout<<"Number of zCandidate with both daughters with number of muon hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauNofMuonHitsLower1","zMassOneDauNofMuonHitsLower1" , 200, 0, 200);
  Events.Project("zMassOneDauNofMuonHitsLower1", "zGoldenMass", cut_zGolden + "zGoldenDau1NofMuonHits==0 || zGoldenDau2NofMuonHits==0"  );
  cout<<"Number of zCandidate with at least one daughter with number of muon hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;


  // Number of  Muon Hits for outer track
  dir->cd();
  // caveat: I'm  requiring isolations
  TH1F * dauSaNofMuonHits = new TH1F("dauSaNofMuonHits", "dauSaNofMuonHits", 100, 0, 100);
  TH1F * h2 = new TH1F("h2", "h2", 100, 0, 100);
  Events.Project("dauSaNofMuonHits", "zGoldenDau1SaNofMuonHits", cut_zGolden );
  Events.Project("h2", "zGoldenDau2SaNofMuonHits", cut_zGolden );
  dauSaNofMuonHits->Add(h2);
  dauSaNofMuonHits->Write();
  delete dauSaNofMuonHits;
  delete h2;



  dir->cd();  
  TH1F * zMass = new TH1F("zMassBothDauSaNofMuonHitsLower1", "zMass", 200, 0, 200);
  Events.Project("zMassBothDauSaNofMuonHitsLower1", "zGoldenMass", cut_zGolden + "zGoldenDau1SaNofMuonHits==0 && zGoldenDau2SaNofMuonHits==0"  );
  cout<<"Number of zCandidate with both daughters->outerTrack with number of muon hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauSaNofMuonHitsLower1","zMassOneDauSaNofMuonHitsLower1" , 200, 0, 200);
  Events.Project("zMassOneDauSaNofMuonHitsLower1", "zGoldenMass", cut_zGolden + "zGoldenDau1SaNofMuonHits==0 || zGoldenDau2SaNofMuonHits==0"  );
  cout<<"Number of zCandidate with at least one daughters->outerTrack with number of muon hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;


 



  // dxyFromBS
  dir->cd();
  TH1F * zMass = new TH1F("zMassBothDaudxyFromBSHigher0_2","zMassBothDaudxyFromBSHigher0_2" , 200, 0, 200);
  Events.Project("zMassBothDaudxyFromBSHigher0_2", "zGoldenMass", cut_zGolden + "zGoldenDau1dxyFromBS>0.2 && zGoldenDau2dxyFromBS>0.2"  );
  cout<<"Number of zCandidate with both daughters with dxyFromBS higher than 0.2: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDaudxyFromBSHigher0_2", "zMassOneDaudxyFromBSHigher0_2", 200, 0, 200);
  Events.Project("zMassOneDaudxyFromBSHigher0_2", "zGoldenMass", cut_zGolden + "zGoldenDau1dxyFromBS>0.2 || zGoldenDau2dxyFromBS>0.2"  );
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
  TH1F * zEta = new TH1F("zEtaBothDauNofMuonHitsLower1","zEtaBothDauNofMuonHitsLower1" , 200, -3, 3);
  Events.Project("zEtaBothDauNofMuonHitsLower1", "zGoldenEta", cut_zGolden + "zGoldenDau1NofMuonHits==0 && zGoldenDau2NofMuonHits==0"  );
  setGraphics(zEta);
  zEta->Write();
  delete zEta;

  dir->cd();
  TH1F * zEta = new TH1F("zEtaOneDauNofMuonHitsLower1","zEtaOneDauNofMuonHitsLower1" , 200, -3, 3);
  Events.Project("zEtaOneDauNofMuonHitsLower1", "zGoldenEta", cut_zGolden + "zGoldenDau1NofMuonHits==0 || zGoldenDau2NofMuonHits==0"  );
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

  
 

  //   output_file->Close(); 
 
}




void qualityStudiesZGlbTrk(TFile *output_file){

  gStyle->SetOptStat();
  gROOT->SetStyle("Plain");
  using namespace std;

  //  #include <exception>;

  //TFile *file = TFile::Open("rfio:/castor/cern.ch/user/d/degrutto/incl15WithBsPv/NtupleLoose_test_inclu15_1_2.root");
  // TFile *file = TFile::Open("../NutpleLooseTestNew_oneshot_all_10_1.root");
//TFile *file = TFile::Open("rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/NtupleLoose_test.root");
//    TTree * Events = dynamic_cast< TTree *> (file->Get("Events"));

  TChain Events("Events"); 
  
  Events.Add("NtupleLooseTestNew_oneshot_all_Test_1_None.root");
 Events.Add("NtupleLooseTestNew_oneshot_all_Test_2_None.root");
 Events.Add("NtupleLooseTestNew_oneshot_all_Test_6_None.root");
  //  Events.Add("../NutpleLooseTestNew_oneshot_all_11_1.root");
  //  Events.Add("../NutpleLooseTestNew_oneshot_all_12_1.root");
  //Events.Add("../NutpleLooseTestNew_oneshot_all_13_1.root");
  //Events.Add("../NutpleLooseTestNew_oneshot_all_4_1.root");
  //Events.Add("../NutpleLooseTestNew_oneshot_all_5_1.root");
  //Events.Add("../NutpleLooseTestNew_oneshot_all_6_1.root");
  //Events.Add("../NutpleLooseTestNew_oneshot_all_7_1.root");
  //Events.Add("../NutpleLooseTestNew_oneshot_all_8_1.root");
  
 //   TFile * output_file = TFile::Open("histo.root", "OPEN");
  // TFile * output_file = TFile::Open("histo_test.root", "RECREATE");
  
  // zMuTrk plots
  TCut cut_zMuTrk("zMuTrkMass>60 && zMuTrkMass<120 && zMuTrkDau1Pt> 20 && zMuTrkDau2Pt>20 && zMuTrkDau1Iso< 3.0 && zMuTrkDau2Iso < 3.0 && zMuTrkDau1Eta<2.1 &&  zMuTrkDau2Eta<2.1");
  TDirectory * dir = output_file->mkdir("goodZToMuMuOneTrackPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  //  Events.Draw("zMuTrkMass");
  Events.Project("zMass", "zMuTrkMass", cut_zMuTrk );
  cout<<"Number of zMuTrk : "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  //Quality checks


 // Chi2

 // quality variables
  // caveat: I'm  requiring isolations
  TH1F * dauChi2 = new TH1F("duaChi2", "dauChi2", 1000, 0, 100);
  // TH1F * h2 = new TH1F("h2", "h2", 1000, 0, 100);
  Events.Project("duaChi2", "zMuTrkDau1Chi2", cut_zMuTrk );
  // Events.Project("h2", "zMuTrkDau2Chi2", cut_zMuTrk );
  //dauChi2->Add(h2);
  dauChi2->Write();
  delete dauChi2;
 
   

  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauChi2Higher10", "zMassOneDauChi2Higher10", 200, 0, 200);
  //  Events.Draw("zMuTrkMass");
  Events.Project("zMassOneDauChi2Higher10", "zMuTrkMass", cut_zMuTrk +"zMuTrkDau1TrkChi2>10");
  setGraphics(zMass);
  zMass->Write();
  cout<<"Number of zCandidate with the global daughter with Chi2 higher than 10: "<<zMass->GetEntries()<<endl;
  delete zMass;

 

  dir->cd();
  TH1F * zMass = new TH1F("zChi2NofMuonHits0","zChi2NofMuonHits0" , 200, 0, 200);
  //  Events.Draw("zMuTrkMass");
  Events.Project("zChi2NofMuonHits0", "zMuTrkDau1Chi2", cut_zMuTrk +  "zMuTrkDau1NofMuonHits==0");
  setGraphics(zMass);
  zMass->Write();
  //cout<<"Number of zCandidate with at least one daughter with Chi2 higher: "<<zMass->GetEntries()<<endl;
  delete zMass;

  dir->cd();
  TH1F * zMass = new TH1F("zChi2NofStripsHits0", "zChi2NofStripsHits0", 200, 0, 20);
  //  Events.Draw("zMuTrkMass");
  Events.Project("zChi2NofStripsHits0", "zMuTrkDau1Chi2", cut_zMuTrk +  "zMuTrkDau1TrkNofStripHits<10");
  setGraphics(zMass);
  zMass->Write();
  //cout<<"Number of zCandidate with at least one daughter with Chi2 higher: "<<zMass->GetEntries()<<endl;
  delete zMass;


  dir->cd();
  TH1F * zMass = new TH1F("zSaChi2NofMuonHits0","zSaChi2NofMuonHits0" , 200, 0, 200);
  //  Events.Draw("zMuTrkMass");
  Events.Project("zSaChi2NofMuonHits0", "zMuTrkDau1SaChi2", cut_zMuTrk +  "zMuTrkDau1NofMuonHits==0");
  setGraphics(zMass);
  zMass->Write();
  //cout<<"Number of zCandidate with at least one daughter with Chi2 higher: "<<zMass->GetEntries()<<endl;
  delete zMass;

  
 

 // Number of  Strips Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  TH1F * dauNofStripHits = new TH1F("dauNofStripHits", "dauNofStripHits", 100, 0, 100);
  Events.Project("dauNofStripHits", "zMuTrkDau1TrkNofStripHits", cut_zMuTrk );
  dauNofStripHits->Write();
  delete dauNofStripHits;

 

  dir->cd();
  TH1F * zMass = new TH1F("zMassBothDauNofStripsHitsLower10","zMassBothDauNofStripsHitsLower10" , 200, 0, 200);
  Events.Project("zMassBothDauNofStripsHitsLower10", "zMuTrkMass", cut_zMuTrk +"zMuTrkDau1TrkNofStripHits<10");
  cout<<"Number of zCandidate with the global daughters with number of strips hits lower than 10: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

// Number of  Strips Hits for inner track
  dir->cd();
  // caveat: I'm  requiring isolations
  TH1F * dauTrkNofStripHits = new TH1F("dauTrkNofStripHits", "dauTrkNofStripHits", 100, 0, 100);
  Events.Project("dauTrkNofStripHits", "zMuTrkDau1TrkNofStripHits", cut_zMuTrk );
  dauTrkNofStripHits->Write();
  delete dauTrkNofStripHits;
 
 

  dir->cd();
  TH1F * zMass = new TH1F("zMassBothDauTrkNofStripsHitsLower10","zMassBothDauTrkNofStripsHitsLower10" , 200, 0, 200);
  Events.Project("zMassBothDauTrkNofStripsHitsLower10", "zMuTrkMass", cut_zMuTrk +"zMuTrkDau1TrkNofStripHits<10");
  cout<<"Number of zCandidate with global daughters->innerTrack() with number of strips hits lower than 10: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;



 // Number of  Pixel Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  TH1F * dauNofPixelHits = new TH1F("dauNofPixelHits", "dauNofPixelHits", 100, 0, 100);
  Events.Project("dauNofPixelHits", "zMuTrkDau1TrkNofPixelHits", cut_zMuTrk );
  dauNofPixelHits->Write();
  delete dauNofPixelHits;
    
  
 // Number of  Pixel Hits
  dir->cd();  
  TH1F * zMass = new TH1F("zMassBothDauNofPixelHitsLower1","zMassBothDauNofPixelHitsLower1" , 200, 0, 200);
  //  Events.Project("zMassBothDauNofPixelHitsLower1", "zMuTrkMass", cut_zMuTrk + "zMuTrkDau1NofPixelHits==0 && (zMuTrkDau2NofPixelHits==0 &&  zMuTrkDau1Chi2<10 || zMuTrkDau2Chi2<10)"  );
  Events.Project("zMassBothDauNofPixelHitsLower1", "zMuTrkMass", cut_zMuTrk + "zMuTrkDau1TrkNofPixelHits==0"  );
  cout<<"Number of zCandidate with global daughter with number of pixel hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

 // Number of  Pixel Hits for inner track
  dir->cd();
  // caveat: I'm  requiring isolations
  TH1F * dauTrkNofPixelHits = new TH1F("dauTrkNofPixelHits", "dauTrkNofPixelHits", 100, 0, 100);
  Events.Project("dauTrkNofPixelHits", "zMuTrkDau1TrkNofPixelHits", cut_zMuTrk );
  dauTrkNofPixelHits->Write();
  delete dauTrkNofPixelHits;
    
  
 // Number of  Pixel Hits
  dir->cd();  
  TH1F * zMass = new TH1F("zMassBothDauTrkNofPixelHitsLower1","zMassBothDauTrkNofPixelHitsLower1" , 200, 0, 200);
  //  Events.Project("zMassBothDauTrkNofPixelHitsLower1", "zMuTrkMass", cut_zMuTrk + "zMuTrkDauTrk1NofPixelHits==0 && (zMuTrkDauTrk2NofPixelHits==0 &&  zMuTrkDauTrk1Chi2<10 || zMuTrkDauTrk2Chi2<10)"  );
  Events.Project("zMassBothDauTrkNofPixelHitsLower1", "zMuTrkMass", cut_zMuTrk + "zMuTrkDau1TrkNofPixelHits==0"  );
  cout<<"Number of zCandidate with global daughter->innerTrack() with number of pixel hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;



  // Number of  Muon Hits for global track
  dir->cd();
  // caveat: I'm  requiring isolations
  TH1F * dauNofMuonHits = new TH1F("dauNofMuonHits", "dauNofMuonHits", 100, 0, 100);
  Events.Project("dauNofMuonHits", "zMuTrkDau1NofMuonHits", cut_zMuTrk );
  dauNofMuonHits->Write();
  delete dauNofMuonHits;




  dir->cd();  
  TH1F * zMass = new TH1F("zMassBothDauNofMuonHitsLower1", "zMass", 200, 0, 200);
  Events.Project("zMassBothDauNofMuonHitsLower1", "zMuTrkMass", cut_zMuTrk + "zMuTrkDau1NofMuonHits==0 "  );
  cout<<"Number of zCandidate with global daughter with number of muon hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  // Number of  Muon Hits for outer track
  dir->cd();
  // caveat: I'm  requiring isolations
  TH1F * dauSaNofMuonHits = new TH1F("dauSaNofMuonHits", "dauSaNofMuonHits", 100, 0, 100);
  Events.Project("dauSaNofMuonHits", "zMuTrkDau1SaNofMuonHits", cut_zMuTrk );
  dauSaNofMuonHits->Write();
  delete dauSaNofMuonHits;




  dir->cd();  
  TH1F * zMass = new TH1F("zMassBothDauSaNofMuonHitsLower1", "zMass", 200, 0, 200);
  Events.Project("zMassBothDauSaNofMuonHitsLower1", "zMuTrkMass", cut_zMuTrk + "zMuTrkDau1SaNofMuonHits==0"  );
  cout<<"Number of zCandidate with global daughter->outerTrack with number of muon hits equal to zero: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

 



  // dxyFromBS
  dir->cd();
  TH1F * zMass = new TH1F("zMassBothDaudxyFromBSHigher0_2","zMassBothDaudxyFromBSHigher0_2" , 200, 0, 200);
  Events.Project("zMassBothDaudxyFromBSHigher0_2", "zMuTrkMass", cut_zMuTrk + "zMuTrkDau1dxyFromBS>0.2 && zMuTrkDau2dxyFromBS>0.2"  );
  cout<<"Number of zCandidate with both daughters with dxyFromBS higher than 0.2: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDaudxyFromBSHigher0_2", "zMassOneDaudxyFromBSHigher0_2", 200, 0, 200);
  Events.Project("zMassOneDaudxyFromBSHigher0_2", "zMuTrkMass", cut_zMuTrk + "zMuTrkDau1dxyFromBS>0.2 || zMuTrkDau2dxyFromBS>0.2"  );
  cout<<"Number of zCandidate with at least one daughter with dxyFromBS higher than 0.2: "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;
 

  // isTrackerMuon
 
  dir->cd();
  TH1F * zMass = new TH1F("zMassOneDauNoTrackerMuon", "zMassOneDauNoTrackerMuon", 200, 0, 200);
  Events.Project("zMassOneDauNoTrackerMuon", "zMuTrkMass", cut_zMuTrk + "zMuTrkDau1TrackerMuonBit==0 "  );
  cout<<"Number of zCandidate with at least one daughter not TrackerMuon : "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;
  


  // Eta distribution if MuonHits is zero
  
  dir->cd();
  TH1F * zEta = new TH1F("zEtaOneDauNofMuonHitsLower1","zEtaOneDauNofMuonHitsLower1" , 200, -3, 3);
  Events.Project("zEtaOneDauNofMuonHitsLower1", "zMuTrkEta", cut_zMuTrk + "zMuTrkDau1NofMuonHits==0 || zMuTrkDau2NofMuonHits==0"  );
  setGraphics(zEta);
  zEta->Write();
  delete zEta;
  

  // Correlation study

  // *** Chi2 vs MuonHits  ***
 
  dir->cd();
  TH2F * Chi2VsMuonHits = new TH2F("Chi2VsMuonHits", "Chi2VsMuonHits", 100, 0, 60, 100, 0, 6);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events.Project("Chi2VsMuonHits", "zMuTrkDau1Chi2:zMuTrkDau1NofMuonHits", cut_zMuTrk);         
  Chi2VsMuonHits->SetDrawOption("Box"); 

  Chi2VsMuonHits->Write();

  delete Chi2VsMuonHits;


  // *** Chi2 vs StripHits  ***
 
  dir->cd();
  TH2F * Chi2VsStripHits = new TH2F("Chi2VsStripHits", "Chi2VsStripHits", 100, 0, 30, 100, 0, 6);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events.Project("Chi2VsStripHits", "zMuTrkDau1Chi2:zMuTrkDau1TrkNofStripHits", cut_zMuTrk);         
  Chi2VsStripHits->SetDrawOption("Box"); 

  Chi2VsStripHits->Write();

  delete Chi2VsStripHits;

   // *** MuonHits vs Eta  ***
 
  dir->cd();
  TH2F * MuonHitsVsEta = new TH2F("MuonHitsVsEta", "MuonHitsVsEta", 100, -2.5, 2.5, 100, 0, 60);
  //TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events.Project("MuonHitsVsEta", "zMuTrkDau1NofMuonHits:zMuTrkDau1Eta", cut_zMuTrk);         
  MuonHitsVsEta->SetDrawOption("Box"); 

  MuonHitsVsEta->Write();

  delete MuonHitsVsEta;


  output_file->cd("/");

  
 


 
}




void qualityStudies(){

TFile * output_file = TFile::Open("histo.root", "RECREATE");
  qualityStudiesZGolden(output_file);
  qualityStudiesZGlbTrk(output_file);
   output_file->Close(); 
}
