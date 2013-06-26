/*

basic comparison plottig 
using ntuples taken from files in ./root/ directory

.x muonDBComparison.C

*/

//#include "TTreeAnalysis.h"

struct AlighnmentTTrees {
  TTree* DTWheels;
  TTree* DTStations;
  TTree* DTChambers;
  TTree* DTSuperLayers;
  TTree* DTLayers;     
  TTree* CSCStations;
  TTree* CSCChambers;
  TTree* CSCLayers;
};

/*
struct AlighnmentTTreesAnalysis {
  TTreeAnalysis DTWheels;
  TTreeAnalysis DTChambers;
  TTreeAnalysis DTStations;
  TTreeAnalysis DTSuperLayers;
  TTreeAnalysis DTLayers;     
  TTreeAnalysis CSCStations;
  TTreeAnalysis CSCChambers;
  TTreeAnalysis CSCLayers;
};
*/

//----------------------------------------------------------------------------------------

void importTreesFromFile(char *filename, TString dirname, AlighnmentTTrees *at, TString namePrf = "")
{
  TFile* f = new TFile(filename);

  TTree* tmpTree_DTWheels      = (TTree*)f->Get((dirname+"/DTWheels").Data());
  TTree* tmpTree_DTStations    = (TTree*)f->Get((dirname+"/DTStations").Data());
  TTree* tmpTree_DTChambers    = (TTree*)f->Get((dirname+"/DTChambers").Data());
//  TTree* tmpTree_DTSuperLayers = (TTree*)f->Get((dirname+"/DTSuperLayers").Data());
//  TTree* tmpTree_DTLayers      = (TTree*)f->Get((dirname+"/DTLayers").Data());

  TTree* tmpTree_CSCStations = (TTree*)f->Get((dirname+"/CSCStations").Data());
  TTree* tmpTree_CSCChambers = (TTree*)f->Get((dirname+"/CSCChambers").Data());
//  TTree* tmpTree_CSCLayers   = (TTree*)f->Get((dirname+"/CSCLayers").Data());

  gDirectory->Cd("Rint:/");
  at->DTWheels	   = (TTree*) tmpTree_DTWheels->CopyTree("1");
  at->DTStations	   = (TTree*) tmpTree_DTStations->CopyTree("1");
  at->DTChambers    = (TTree*) tmpTree_DTChambers->CopyTree("1");
//  at->DTSuperLayers = (TTree*) tmpTree_DTSuperLayers->CopyTree("1");
//  at->DTLayers	   = (TTree*) tmpTree_DTLayers->CopyTree("1");
  
  at->CSCStations = (TTree*) tmpTree_CSCStations->CopyTree("1");
  at->CSCChambers = (TTree*) tmpTree_CSCChambers->CopyTree("1");
//  at->CSCLayers   = (TTree*) tmpTree_CSCLayers->CopyTree("1");
  
  at->DTWheels     ->SetName((namePrf+"DTWheels"     ).Data());
  at->DTStations   ->SetName((namePrf+"DTStations"   ).Data());
  at->DTChambers   ->SetName((namePrf+"DTChambers"   ).Data());
//  at->DTSuperLayers->SetName((namePrf+"DTSuperLayers").Data());
//  at->DTLayers     ->SetName((namePrf+"DTLayers"     ).Data());
  at->CSCStations  ->SetName((namePrf+"CSCStations"  ).Data());
  at->CSCChambers  ->SetName((namePrf+"CSCChambers"  ).Data());
//  at->CSCLayers    ->SetName((namePrf+"CSCLayers"    ).Data());

  f->Close();
}

//----------------------------------------------------------------------------------------

void setupStyle()
{
  gStyle->SetStatBorderSize(1);
  gStyle->SetStatStyle(3001);
  gStyle->SetOptStat(1111110);
  gStyle->SetOptFit(1111);
  gStyle->SetStatColor(0);
  gStyle->SetStatFontSize(0.03);
  gStyle->SetStatW(0.28);
  gStyle->SetStatH(0.32);
  gStyle->SetStatX(.99);
  gStyle->SetStatY(.99);

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadTopMargin(0.13);
  gStyle->SetPadLeftMargin(0.16);
  gStyle->SetPadRightMargin(0.06);
  gStyle->SetTitleYOffset(1.2);
  gStyle->SetTitleXOffset(.65);
  gStyle->SetTitleYSize(0.07);
  gStyle->SetTitleXSize(0.07);
  gStyle->SetLabelSize(0.05);
  gStyle->SetLabelSize(0.05,"Y");

  gStyle->SetTitleBorderSize(0);
  //gStyle->SetTitleStyle(3001);
  gStyle->SetTitleColor(1);
  gStyle->SetTitleFillColor(0);
  gStyle->SetTitleFontSize(0.07);
  gStyle->SetTitleStyle(0);

}

//----------------------------------------------------------------------------------------

void setupHisto(TH1F *h, char title[200], int color=-1, int lcolor=-1)
{
  h->SetTitle(title);
  if (color>=0) h->SetFillColor(color);
  if (lcolor>=0) {
    h->SetLineColor(lcolor);
    h->SetLineWidth(2);
  }
}

//----------------------------------------------------------------------------------------

void adjustRange(TH1F *h, TH1F *h1)
{
  double ymax = (h->GetMaximum() > h1->GetMaximum()) ? h->GetMaximum() : h1->GetMaximum();
  h1->SetMaximum(ymax*1.05);
  //gPad->Update();
}


//----------------------------------------------------------------------------------------

void compare2Alignments(
  char *filename,
  TString alment0="mockAlignment",
  TString alment1="ideal")
{
  setupStyle();
  
  AlighnmentTTrees alignTrees[2];

  importTreesFromFile(filename, alment0, &alignTrees[0], alment0+"_");
  importTreesFromFile(filename, alment1, &alignTrees[1], alment1+"_");


/*
  gSystem->CompileMacro("loadLookupTable.C","k");
  //gSystem->Load("loadLookupTable_C.so");
  loadLookupTable();
//  populateDetData(atrees[0].DTWheels);
  for (int i=0; i<2; i++) {
    populateDetData(atrees[i].DTWheels);
    populateDetData(atrees[i].DTChambers);
    populateDetData(atrees[i].DTStations);
    populateDetData(atrees[i].DTSuperLayers);
    populateDetData(atrees[i].DTLayers);
    populateDetData(atrees[i].CSCStations);
    populateDetData(atrees[i].CSCChambers);
    populateDetData(atrees[i].CSCLayers);
  }
  
  AlighnmentTTreesAnalysis alignTreesAna[2];
  for (int i=0; i<2; i++) {
    alignTreesAna[i].DTWheels.Init(alignTrees[i].DTWheels);
    alignTreesAna[i].DTChambers.Init(alignTrees[i].DTChambers);
    alignTreesAna[i].DTStations.Init(alignTrees[i].DTStations);
    alignTreesAna[i].DTSuperLayers.Init(alignTrees[i].DTSuperLayers);
    alignTreesAna[i].DTLayers.Init(alignTrees[i].DTLayers);
    alignTreesAna[i].CSCStations.Init(alignTrees[i].CSCStations);
    alignTreesAna[i].CSCChambers.Init(alignTrees[i].CSCChambers);
    alignTreesAna[i].CSCLayers.Init(alignTrees[i].CSCLayers);
  }
*/

  alignTrees[0].DTWheels->AddFriend(alignTrees[1].DTWheels,"a");
  alignTrees[0].DTChambers->AddFriend(alignTrees[1].DTChambers,"a");
  alignTrees[0].DTStations->AddFriend(alignTrees[1].DTStations,"a");
//  alignTrees[0].DTSuperLayers->AddFriend(alignTrees[1].DTSuperLayers,"a");
//  alignTrees[0].DTLayers->AddFriend(alignTrees[1].DTLayers,"a");
  alignTrees[0].CSCStations->AddFriend(alignTrees[1].CSCStations,"a");
  alignTrees[0].CSCChambers->AddFriend(alignTrees[1].CSCChambers,"a");
//  alignTrees[0].CSCLayers->AddFriend(alignTrees[1].CSCLayers,"a");

  /// ------ PLOTTING -------------

  char histostr[100], drawstr[300], cutstr[200], hsetupstr[200];
  //double nbin = 20, low = -0.15, high = 0.15;
  double nbin = 20, low = -0.05, high = 0.05;
  
  TCanvas *c1 = new TCanvas("c_barrel_dz","c_barrel_dz",1500,350);
  c1->Divide(5,1,0.001,0.001);
  TH1F *h_bz_[5];
  for (int i=-2; i<=2; i++) {
    c1->cd(i+3);
    sprintf(histostr,"h_bz_[%d]",i+2);
    //sprintf(drawstr,"z-a.z>>%s(%d,%f,%f)",histostr,nbin,low,high);
    sprintf(drawstr,"xhatx*dx+xhaty*dy+xhatz*dz>>%s(%d,%f,%f)",histostr,nbin,low,high);
    sprintf(cutstr, "structa==%d",i);
    //sprintf(hsetupstr,"wheel %d:  #Delta z;#Delta z;# chambers", i);
    sprintf(hsetupstr,"wheel %d:  #Delta x;#Delta x;# chambers", i);
    cout<<drawstr<<endl;
    alignTrees[0].DTChambers->Draw(drawstr,cutstr);
    h_bz_[i+2] = (TH1F*) gROOT->FindObject(histostr);
    setupHisto(h_bz_[i+2],hsetupstr, 3);
    h_bz_[i+2]->GetXaxis()->SetNdivisions(207);
  }
  c1->cd();

  
  //nbin = 20; low = -1.5; high = 1.5;
  nbin = 20; low = -.02; high = .02;
  
  TCanvas *c2 = new TCanvas("c_endcap_dz","c_endcap_dz",1200,700);
  c2->Divide(4,2,0.001,0.001);
  TH1F *h_edz_[9];
  int npad=1;
  for (int i=-4; i<=4; i++) {
    if (i==0) continue;
    c2->cd(npad++);
    sprintf(histostr,"h_edz_[%d]",i+4);
    //sprintf(drawstr,"z-a.z>>%s(%d,%f,%f)",histostr,nbin,low,high);
    sprintf(drawstr,"xhatx*dx+xhaty*dy+xhatz*dz>>%s(%d,%f,%f)",histostr,nbin,low,high);
    //sprintf(cutstr, "structa==%d",i);
    sprintf(cutstr, "structa==%d&&(!(structa==1&&(structb==1||structb==4)))",i);
    //sprintf(hsetupstr,"disk %d:  #Delta z;#Delta z;# chambers", i);
    sprintf(hsetupstr,"disk %d:  #Delta x;#Delta x;# chambers", i);
    cout<<drawstr<<endl;
    alignTrees[0].CSCChambers->Draw(drawstr,cutstr);
    h_edz_[i+4] = (TH1F*) gROOT->FindObject(histostr);
    setupHisto(h_edz_[i+4],hsetupstr, 3);
    h_edz_[i+4]->GetXaxis()->SetNdivisions(207);
  }
  c2->cd();


  //TCanvas *tc = new TCanvas("test","test",900,900);
  //tc->Divide(2,2);
  //tc->cd(1);


  /// ------ END PLOTTING -------------
  
  return;
 
  if (gROOT->IsBatch()) return;
  new TBrowser();
  TTreeViewer *treeview = new TTreeViewer();
//  char tnames[8][50] = {"DTWheels","DTStations","DTChambers","DTSuperLayers","DTLayers" ,"CSCStations" ,"CSCChambers" ,"CSCLayers"};
  char tnames[8][50] = {"DTWheels","DTStations","DTChambers","CSCStations" ,"CSCChambers" };
  for (int i=0;i<8;i++)  {
    char nm[200];
    sprintf(nm,"%s_%s",alment0.Data(),tnames[i]);
    treeview->SetTreeName(nm);
  }
  for (int i=0;i<8;i++)  {
    char nm[200];
    sprintf(nm,"%s_%s",alment1.Data(),tnames[i]);
    treeview->SetTreeName(nm);
  }


}



//----------------------------------------------------------------------------------------

void compare2AlignmentsSame(
  char *filename,
  TString alment0="fromAlignment",
  TString alment1="ideal")
{
  setupStyle();
  
  AlighnmentTTrees alignTrees[2];

  importTreesFromFile(filename, alment0, &alignTrees[0], alment0+"_");
  importTreesFromFile(filename, alment1, &alignTrees[1], alment1+"_");


  alignTrees[0].DTWheels->AddFriend(alignTrees[1].DTWheels,"a");
  alignTrees[0].DTChambers->AddFriend(alignTrees[1].DTChambers,"a");
  alignTrees[0].DTStations->AddFriend(alignTrees[1].DTStations,"a");
//  alignTrees[0].DTSuperLayers->AddFriend(alignTrees[1].DTSuperLayers,"a");
//  alignTrees[0].DTLayers->AddFriend(alignTrees[1].DTLayers,"a");
  alignTrees[0].CSCStations->AddFriend(alignTrees[1].CSCStations,"a");
  alignTrees[0].CSCChambers->AddFriend(alignTrees[1].CSCChambers,"a");
//  alignTrees[0].CSCLayers->AddFriend(alignTrees[1].CSCLayers,"a");

  /// ------ PLOTTING -------------

  char histostr[100], histostr1[100], drawstr[300], cutstr[200], hsetupstr[200];
  //double nbin = 20, low = -0.15, high = 0.15;
  double nbin = 20, low = -0.05, high = 0.05;
  
  TCanvas *c1 = (TCanvas*)gROOT->FindObject("c_barrel_dz");
  if (c1==NULL) return;
  TH1F *h_1;
  TH1F *h_bz_[5];
  for (int i=-2; i<=2; i++) {
    c1->cd(i+3);
    sprintf(histostr,"h_bz_s[%d]",i+2);
    sprintf(histostr1,"h_bz_[%d]",i+2);
    //sprintf(drawstr,"z-a.z>>%s(%d,%f,%f)",histostr,nbin,low,high);
    sprintf(drawstr,"xhatx*dx+xhaty*dy+xhatz*dz>>%s(%d,%f,%f)",histostr,nbin,low,high);
    sprintf(cutstr, "structa==%d",i);
    //sprintf(hsetupstr,"wheel %d:  #Delta z;#Delta z;# chambers", i);
    sprintf(hsetupstr,"wheel %d:  #Delta x;#Delta x;# chambers", i);
    cout<<drawstr<<endl;
    alignTrees[0].DTChambers->Draw(drawstr,cutstr,"same");
    h_bz_[i+2] = (TH1F*) gROOT->FindObject(histostr);
    setupHisto(h_bz_[i+2],hsetupstr, -1, 4);
    h_1 = (TH1F*) gROOT->FindObjectAny(histostr1);
    adjustRange(h_bz_[i+2],h_1);
  }
  c1->cd();

  
  //nbin = 20; low = -1.5; high = 1.5;
  nbin = 20; low = -.02; high = .02;
  
  TCanvas *c2 = (TCanvas*)gROOT->FindObject("c_endcap_dz");
  if (c2==NULL) return;
  TH1F *h_edz_[9];
  int npad=1;
  for (int i=-4; i<=4; i++) {
    if (i==0) continue;
    c2->cd(npad++);
    sprintf(histostr,"h_edz_s[%d]",i+4);
    sprintf(histostr1,"h_edz_[%d]",i+4);
    //sprintf(drawstr,"z-a.z>>%s(%d,%f,%f)",histostr,nbin,low,high);
    sprintf(drawstr,"xhatx*dx+xhaty*dy+xhatz*dz>>%s(%d,%f,%f)",histostr,nbin,low,high);
    //sprintf(cutstr, "structa==%d",i);
    sprintf(cutstr, "structa==%d&&(!(abs(structa)==1&&(structb==1||structb==4)))",i);
    //sprintf(hsetupstr,"disk %d:  #Delta z;#Delta z;# chambers", i);
    sprintf(hsetupstr,"disk %d:  #Delta x;#Delta x;# chambers", i);
    cout<<drawstr<<endl;
    alignTrees[0].CSCChambers->Draw(drawstr,cutstr,"same");
    h_edz_[i+4] = (TH1F*) gROOT->FindObject(histostr);
    setupHisto(h_edz_[i+4],hsetupstr, -1, 4);
    h_1 = (TH1F*) gROOT->FindObjectAny(histostr1);
    adjustRange(h_edz_[i+4],h_1);
  }
  c2->cd();


  //TCanvas *tc = new TCanvas("test","test",900,900);
  //tc->Divide(2,2);
  //tc->cd(1);


  /// ------ END PLOTTING -------------
  
  return;
 
  if (gROOT->IsBatch()) return;
  new TBrowser();
  TTreeViewer *treeview = new TTreeViewer();
//  char tnames[8][50] = {"DTWheels","DTStations","DTChambers","DTSuperLayers","DTLayers" ,"CSCStations" ,"CSCChambers" ,"CSCLayers"};
  char tnames[8][50] = {"DTWheels","DTStations","DTChambers","CSCStations" ,"CSCChambers" };
  for (int i=0;i<8;i++)  {
    char nm[200];
    sprintf(nm,"%s_%s",alment0.Data(),tnames[i]);
    treeview->SetTreeName(nm);
  }
  for (int i=0;i<8;i++)  {
    char nm[200];
    sprintf(nm,"%s_%s",alment1.Data(),tnames[i]);
    treeview->SetTreeName(nm);
  }


}



//----------------------------------------------------------------------------------------

void compareAlignmentsSeries(
  int N, 
  char files[10][255],
  TString alment0,
  TString alment)
{
//  gSystem->Load("TTreeAnalysis_C.so");

  setupStyle();
  
  AlighnmentTTrees alignTrees[10];

  char dum[100];
  sprintf(dum,"%s_",alment0.Data());
  importTreesFromFile(files[0], alment0, &alignTrees[0], alment0+"_");
  for (Int_t i=1; i<=N; i++) {
    sprintf(dum,"%s_%d_",alment.Data(),i);
    importTreesFromFile(files[i-1], alment, &alignTrees[i], dum);

    alignTrees[i].DTWheels->AddFriend(alignTrees[0].DTWheels,"a");
    alignTrees[i].DTChambers->AddFriend(alignTrees[0].DTChambers,"a");
    alignTrees[i].DTStations->AddFriend(alignTrees[0].DTStations,"a");
    //alignTrees[i].DTSuperLayers->AddFriend(alignTrees[0].DTSuperLayers,"a");
    //alignTrees[i].DTLayers->AddFriend(alignTrees[0].DTLayers,"a");
    alignTrees[i].CSCStations->AddFriend(alignTrees[0].CSCStations,"a");
    alignTrees[i].CSCChambers->AddFriend(alignTrees[0].CSCChambers,"a");
    //alignTrees[i].CSCLayers->AddFriend(alignTrees[0].CSCLayers,"a");
  }
  
  /// ------ PLOTTING -------------

  char histostr[100], drawstr[300], cutstr[200], hsetupstr[200];
  double nbin = 20, low = -0.15, high = 0.15;
  
  TCanvas *tc = new TCanvas("test","test",500,500);

//  alignTrees[5].DTChambers->Draw(
//    "atan2(y,x)-atan2(a.y,a.x)>>h_dphi(100)","structa==2");


  TCanvas *c1 = new TCanvas("c_barrel_dxloc","c_barrel_dxloc",1500,350);
  c1->Divide(5,1,0.001,0.001);
  TH1F *h_bdxloc_[6];
  h_bdxloc_[5] = new TH1F("hzero","hzero",5,0.5,5.5);
  h_bdxloc_[5]->SetStats(0);
  h_bdxloc_[5]->SetLineColor(4);
  for (int i=-2; i<=2; i++) {
    sprintf(histostr,"h_bdxloc_[%d]",i+2);
    cout<<histostr<<endl;
    h_bdxloc_[i+2] = new TH1F(histostr,histostr,5,0.5,5.5);
    sprintf(cutstr, "structa==%d",i);
    sprintf(hsetupstr,"wheel %d:  #Delta x;alignment #;#Delta x", i);
    tc->cd();
    for (int f=1; f<=5; f++) {
      alignTrees[f].DTChambers->Draw("xhatx*dx+xhaty*dy+xhatz*dz",cutstr );
//        "sqrt(x*x+y*y)*atan2(sin(atan2(y,x)-atan2(a.y,a.x)),cos(atan2(y,x)-atan2(a.y,a.x)))");
//        "atan2(sin(atan2(y,x)-atan2(a.y,a.x)),cos(atan2(y,x)-atan2(a.y,a.x)))",cutstr);
      h_bdxloc_[i+2]->SetBinContent(f,htemp->GetMean());
      h_bdxloc_[i+2]->SetBinError(f,htemp->GetRMS());
    }
    h_bdxloc_[i+2]->SetEntries(5);
    setupHisto(h_bdxloc_[i+2],hsetupstr, 8);
    c1->cd(i+3);
    h_bdxloc_[i+2]->Draw();
    h_bdxloc_[5]->Draw("same");
  }
  c1->cd();

//return;

  TCanvas *c2 = new TCanvas("c_endcap_dxloc","c_endcap_dxloc",1200,700);
  c2->Divide(4,2,0.001,0.001);
  TH1F *h_edxloc_[10];
  h_edxloc_[9] = new TH1F("hezero","hezero",5,0.5,5.5);
  h_edxloc_[9]->SetStats(0);
  h_edxloc_[9]->SetLineColor(4);
  int npad=1;
  for (int i=-4; i<=4; i++) {
    if (i==0) continue;
    sprintf(histostr,"h_edxloc_[%d]",i+4);
    cout<<histostr<<endl;
    h_edxloc_[i+4] = new TH1F(histostr,histostr,5,0.5,5.5);
    //sprintf(cutstr, "structa==%d&&structb==1",i);
    sprintf(cutstr, "structa==%d",i);
    sprintf(hsetupstr,"disk %d:  #Delta x;alignment #;#Delta x", i);
    tc->cd();
    for (int f=1; f<=5; f++) {
      alignTrees[f].CSCChambers->Draw("xhatx*dx+xhaty*dy+xhatz*dz",cutstr );
//        "sqrt(x*x+y*y)*atan2(sin(atan2(y,x)-atan2(a.y,a.x)),cos(atan2(y,x)-atan2(a.y,a.x)))",cutstr);
//        "atan2(sin(atan2(y,x)-atan2(a.y,a.x)),cos(atan2(y,x)-atan2(a.y,a.x)))",cutstr);
      h_edxloc_[i+4]->SetBinContent(f,htemp->GetMean());
      h_edxloc_[i+4]->SetBinError(f,htemp->GetRMS());
    }
    h_edxloc_[i+4]->SetEntries(5);
    setupHisto(h_edxloc_[i+4],hsetupstr, 8);
    c2->cd(npad++);
    h_edxloc_[i+4]->Draw();
    h_edxloc_[9]->Draw("same");
  }
  c2->cd();
  //c2->Print("r1.ps");
  


  //TCanvas *tc = new TCanvas("test","test",900,900);
  //tc->Divide(2,2);
  //tc->cd(1);


  /// ------ END PLOTTING -------------
  
  return;
 
  if (gROOT->IsBatch()) return;
  new TBrowser();
  TTreeViewer *treeview = new TTreeViewer();
//  char tnames[8][50] = {"DTWheels","DTStations","DTChambers","DTSuperLayers","DTLayers" ,"CSCStations" ,"CSCChambers" ,"CSCLayers"};
  char tnames[8][50] = {"DTWheels","DTStations","DTChambers","CSCStations" ,"CSCChambers" };
  for (int i=0;i<8;i++)  {
    char nm[200];
    sprintf(nm,"%s_%s",alment0.Data(),tnames[i]);
    treeview->SetTreeName(nm);
  }
  for (int i=0;i<8;i++)  {
    char nm[200];
    sprintf(nm,"%s_%s",alment.Data(),tnames[i]);
    treeview->SetTreeName(nm);
  }


}


//----------------------------------------------------------------------------------------

void muonDBComparison()
{

  //  files with series of alignments
//  char files[5][255]={"root/roll0.root", "root/roll2.root", "root/roll4.root", "root/roll6.root", "root/roll8.root"};
  char files[5][255]={"root/tracker0.root", "root/tracker1.root", "root/tracker2.root", "root/tracker3.root", "root/tracker5.root"};

  // draw the series comparison
  compareAlignmentsSeries(5, files,"ideal","fromAlignment");


  // compare mock and ideal 
  compare2Alignments("root/first_examples.root", "mockAlignment", "ideal");

  // draw the same plots for real alignment overlayed
  compare2AlignmentsSame("root/first_examples.root", "fromAlignment","ideal");

}
