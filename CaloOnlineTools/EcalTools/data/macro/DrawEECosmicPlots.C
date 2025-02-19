
void DrawEECosmicPlots(Char_t* infile = 0, Int_t runNum=0, Bool_t doEEPlus=kTRUE, Bool_t printPics = kTRUE, Char_t* fileType = "png", Char_t* dirName = ".", Bool_t doWait=kFALSE)
{

  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1,0); gStyle->SetOptStat(10);

  if (!infile) {
    cout << " No input file specified !" << endl;
    return;
  }

  cout << "Producing cosmics plots for: " << infile << endl;

  TFile* f = new TFile(infile);

  int runNumber = 0;
  if (runNum==0) {
    runNumber = runNumberHist->GetBinContent(1);
    cout << "Run Number: " << runNumber << endl;
  } else {
    runNumber = runNum;
  }

  char name[100];  

  char runChar[50];
  sprintf(runChar,", run %i",runNumber);


  char hname[200];  char fname[200]; char dname[200]; char sname[200];

  if (doEEPlus) {
    sprintf(dname,"EEPlus");      
    sprintf(sname,"EEP");      
  } else {
    sprintf(dname,"EEMinus");      
    sprintf(sname,"EEM");      
  } 

  // First for EE-

  // occupancy plots

  const int nHists1 = 5+4+4+4+4+4;

  char* occupPlots[nHists1] = {
    "OccupancyAllEvents",
    "OccupancyAllEventsCoarse",
    "OccupancySingleXtal",
    "OccupancyHighEnergyEvents",
    "OccupancyHighEnergyEventsCoarse",
    "OccupancyAllEventsCoarse_ECAL",
    "OccupancyAllEvents_ECAL",
    "OccupancyAllEventsCoarse_ExclusiveECAL",
    "OccupancyAllEvents_ExclusiveHCAL",
    "OccupancyAllEventsCoarse_HCAL",
    "OccupancyAllEvents_HCAL",
    "OccupancyAllEventsCoarse_ExclusiveHCAL",
    "OccupancyAllEvents_ExclusiveHCAL",
    "OccupancyAllEventsCoarse_DT",
    "OccupancyAllEvents_DT",
    "OccupancyAllEventsCoarse_ExclusiveDT",
    "OccupancyAllEvents_ExclusiveDT",
    "OccupancyAllEventsCoarse_RPC",
    "OccupancyAllEvents_RPC",
    "OccupancyAllEventsCoarse_ExclusiveRPC",
    "OccupancyAllEvents_ExclusiveRPC",
    "OccupancyAllEventsCoarse_CSC",
    "OccupancyAllEvents_CSC",
    "OccupancyAllEventsCoarse_ExclusiveCSC",
    "OccupancyAllEvents_ExclusiveCSC"
  };

  TCanvas* c1[nHists1];
  char cname[100]; 

  for (int i=0; i<nHists1; i++) {
    sprintf(cname,"c1_%i",i);
    int x = (i%3)*500;     //int x = (i%3)*600;
    int y = (i/3)*100;     //int y = (i/3)*200;
    c1[i] =  new TCanvas(cname,cname,x,y,500,500);
    //    cout << "Hists " << i << " : " << x << " , " << y << endl;
  }


  cout << nHists1 << endl;
  for (int ic=0; ic<nHists1; ic++) {
    cout << ic << endl;
    c1[ic]->cd();
    sprintf(fname,occupPlots[ic]);  
    draw2D(10,fname,dname,c1[ic],f,runNum);
    c1[ic]->SetLogy(0);
    c1[ic]->SetLogz(1);
    drawEELines();
    if (printPics) { sprintf(name,"%s/cosmicsAnalysis_%s_%s_%i.%s",dirName,fname,sname,runNumber,fileType); c1[ic]->Print(name); }  
  }
  

  // 1D Log plots  
  // energies
  const int nHists2 = 6;
  char* oneDimLogPlots[nHists2] = {
    "SeedEnergyAllFEDs",
     "E2_AllClusters",
    "energy_AllClusters",
    "energyHigh_AllClusters",
    "energyHigh_HighEnergyClusters",
    "energy_SingleXtalClusters"
  };

  TCanvas* c2[nHists2];

  for (int i=0; i<nHists2; i++) {
    sprintf(cname,"c2_%i",i);
    int x = ((i)%3)*600;     //int x = (i%3)*600;
    int y = ((i)/3)*100;     //int y = (i/3)*200;
    c2[i] =  new TCanvas(cname,cname,x,y,600,400);
  }
  
  cout << nHists1 << " " << nHists1+nHists2 << endl;
  for (int ic=0; ic<nHists2; ic++) {
    cout << ic << endl;
    c2[ic]->cd();
    sprintf(fname,oneDimLogPlots[ic]);  
    draw1D(10,fname,dname,c2[ic],f,runNum);
    c2[ic]->SetLogy(1);
    if (printPics) { sprintf(name,"%s/cosmicsAnalysis_%s_%s_%i.%s",dirName,fname,sname,runNumber,fileType); c2[ic]->Print(name); }  
  }

  // 1D Lin plots

  const int nHists3 = 13;
  char* oneDimPlots[nHists3] = {
     "NumXtalsInClusterAllHist",
     "NumActiveXtalsInClusterAllHist",
     "numberofBCinSC",
     "numberofCosmicsPerEvent",
     "NumXtalsVsHighEnergy",
     "timeForAllFeds",
     "timeForAllFeds_ECAL",
     "timeForAllFeds_HCAL",
     "timeForAllFeds_DT",
     "timeForAllFeds_RPC",
     "timeForAllFeds_CSC",
     "triggerHist",
     "triggerExclusiveHist" 
  };

  TCanvas* c3[nHists3];

  for (int i=0; i<nHists3; i++) {
    sprintf(cname,"c3_%i",i);
    int x = ((i)%3)*600;     //int x = (i%3)*600;
    int y = ((i)/3)*100;     //int y = (i/3)*200;
    c3[i] =  new TCanvas(cname,cname,x,y,600,400);
  }
  
  for (int ic=0; ic<nHists3; ic++) {
    cout << ic << endl;
    c3[ic]->cd();
    sprintf(fname,oneDimPlots[ic]);  
    draw1D(10,fname,dname,c3[ic],f,runNum);
    c3[ic]->SetLogy(0);
    if (printPics) { sprintf(name,"%s/cosmicsAnalysis_%s_%s_%i.%s",dirName,fname,sname,runNumber,fileType); c3[ic]->Print(name); }  
  }


  //2D

  const int nHists4 = 2;
  char* twoDimPlots[nHists4] = {
    "energyvsE1_AllClusters",
    "timeVsAmpAllEvents"
  };

  TCanvas* c4[nHists4];

  for (int i=0; i<nHists4; i++) {
    sprintf(cname,"c4_%i",i);
    int x = ((i)%3)*600;     //int x = (i%3)*600;
    int y = ((i)/3)*100;     //int y = (i/3)*200;
    c4[i] =  new TCanvas(cname,cname,x,y,600,400);
  }
  
  for (int ic=0; ic<nHists4; ic++) {
    cout << ic << endl;
    c4[ic]->cd();
    sprintf(fname,twoDimPlots[ic]);  
    draw2D(10,fname,dname,c4[ic],f,runNum);
    c4[ic]->SetLogy(0);
    c4[ic]->SetLogx(0);
    c4[ic]->SetLogz(1);
    if (printPics) { sprintf(name,"%s/cosmicsAnalysis_%s_%s_%i.%s",dirName,fname,sname,runNumber,fileType); c4[ic]->Print(name); }  
  }


  // 2D profile

  const int nHists5 = 2;
  char* twoDimProfilePlots[nHists5] = {
    "NumXtalsVsEnergy",
    "NumXtalsVsHighEnergy"
  };

  TCanvas* c5[nHists5];

  for (int i=0; i<nHists5; i++) {
    sprintf(cname,"c5_%i",i);
    int x = ((i)%3)*600;     //int x = (i%3)*600;
    int y = ((i)/3)*100;     //int y = (i/3)*200;
    c5[i] =  new TCanvas(cname,cname,x,y,600,400);
  }
  
  for (int ic=0; ic<nHists5; ic++) {
    cout << ic << endl;
    c5[ic]->cd();
    sprintf(fname,twoDimProfilePlots[ic]);  
    draw2DProfile(10,fname,dname,c5[ic],f,runNum);
    //    c5[ic]->SetLogy(1);
    if (printPics) { sprintf(name,"%s/cosmicsAnalysis_%s_%s_%i.%s",dirName,fname,sname,runNumber,fileType); c5[ic]->Print(name); }  
  }


  // TT binned

  const int nHists6 = 6;
  char* threeDimProjectionPlots[nHists6] = {
    "timeTTAllFEDs",
    "timeTTAllFEDs_ECAL",
    "timeTTAllFEDs_HCAL",
    "timeTTAllFEDs_DT",
    "timeTTAllFEDs_RPC",
    "timeTTAllFEDs_CSC"
  };

  TCanvas* c6[nHists6];

  for (int i=0; i<nHists6; i++) {
    sprintf(cname,"c6_%i",i);
    int x = (i%3)*500;     //int x = (i%3)*600;
    int y = (i/3)*100;     //int y = (i/3)*200;
    c6[i] =  new TCanvas(cname,cname,x,y,500,500);
  }
  
  for (int ic=0; ic<nHists6; ic++) {
    cout << ic << " " << fname << endl;
    c6[ic]->cd();
    sprintf(fname,threeDimProjectionPlots[ic]);  
    draw3DProjection(10,fname,dname,c6[ic],f,runNum);
    c6[ic]->SetLogy(0);
    c6[ic]->SetLogx(0);
    drawEELines();
    if (printPics) { sprintf(name,"%s/cosmicsAnalysis_%s_%s_%i.%s",dirName,fname,sname,runNumber,fileType); c6[ic]->Print(name); }  
  }

  return;
}

void draw1D(int ic, Char_t* fname, Char_t* dname, TCanvas* c, TFile* f, int runNumber) {

  c->cd();
  char runChar[50];  sprintf(runChar,", run %i",runNumber);
  char hname[200]; 

  gStyle->SetOptStat(111110);
  sprintf(hname,"%s/%s",dname,fname);  
  TH1F* h1 = (TH1F*)f->Get(hname);
  h1->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",h1->GetTitle()); 
  strcat(mytitle,runChar); h1->SetTitle(mytitle);
  return;
}

void draw2D(int ic, Char_t* fname, Char_t* dname, TCanvas* c, TFile* f, int runNumber) {
  c->cd();
  char runChar[50];  sprintf(runChar,", run %i",runNumber);
  char hname[200]; 
  
  gStyle->SetOptStat(10);
  //  if (simpleStats) { gStyle->SetOptStat(10); }
  //  else { gStyle->SetOptStat(111110); }
  //  c->Update();

  sprintf(hname,"%s/%s",dname,fname);  
  TH2F* h2 = (TH2F*)f->Get(hname);
  h2->Draw("colz");
  h2->SetMinimum(1);
  char mytitle[100]; sprintf(mytitle,"%s",h2->GetTitle()); 
  strcat(mytitle,runChar); h2->SetTitle(mytitle);
  return;
}

void draw2DProfile(int ic, Char_t* fname, Char_t* dname, TCanvas* c, TFile* f, int runNumber) {

  c->cd();
  char runChar[50];  sprintf(runChar,", run %i",runNumber);
  char hname[200]; 

  gStyle->SetOptStat(10);
  sprintf(hname,"%s/%s",dname,fname);  
  TH2F* h2 = (TH2F*)f->Get(hname);
  h2->ProfileX()->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",h2->GetTitle()); 
  strcat(mytitle,runChar); h2->SetTitle(mytitle);
  return;
}


void draw3DProjection(int ic, Char_t* fname, Char_t* dname, TCanvas* c, TFile* f, int runNumber) {

  gStyle->SetOptStat(10);

  c->cd();
  char runChar[50];  sprintf(runChar,", run %i",runNumber);
  char hname[200]; 

  gStyle->SetOptStat(10);
  sprintf(hname,"%s/%s",dname,fname);  

  TH3F* h3 = (TH3F*)f->Get(hname);
  TProfile2D* ayx = (TProfile2D*) h3->Project3DProfile("yx");
  //  h3->Draw();
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("ix");
  ayx->GetYaxis()->SetTitle("iy");

  char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
  strcat(mytitle,runChar); ayx->SetTitle(mytitle);
  return;

}

void drawEELines() {

  int ixSectorsEE[202] = {61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 0,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100,  0, 61, 65, 65, 70, 70, 80, 80, 90, 90, 92,  0, 61, 65, 65, 90, 90, 97,  0, 57, 60, 60, 65, 65, 70, 70, 75, 75, 80, 80,  0, 50, 50,  0, 43, 40, 40, 35, 35, 30, 30, 25, 25, 20, 20,  0, 39, 35, 35, 10, 10,  3,  0, 39, 35, 35, 30, 30, 20, 20, 10, 10,  8,  0, 45, 45, 40, 40, 35, 35,  0, 55, 55, 60, 60, 65, 65};
 
  int iySectorsEE[202] = {50, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 50,  0, 50, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 50,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 50, 50, 55, 55, 60, 60,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 61,100,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 50, 50, 55, 55, 60, 60,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 39, 30, 30, 15, 15,  5,  0, 39, 30, 30, 15, 15,  5};

 TLine l;
 l.SetLineWidth(1);
 for ( int i=0; i<201; i=i+1) {
   if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
	(ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
     l.DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		ixSectorsEE[i+1], iySectorsEE[i+1]);
   }
 }


}
