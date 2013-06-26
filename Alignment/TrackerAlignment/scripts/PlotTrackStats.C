#include <Riostream.h>
#include <string>
#include <sys/stat.h>

#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TChain.h"
#include "TFile.h"
#include "TDirectory.h"


bool CheckFileExistence(char *filename);
void PlotTrackStats();

void PlotTrackStats(){

  //define paths
  char basepath[128];
  sprintf(basepath,"./MONITORING/DQM");
  char filename[256];

  const int StartFile=1;//211;
  const int FinFile=999;
  int addedfiles=0;

  //create a chain with all the DQM histograms
  TChain *ch=new TChain("AlignmentTrackStats");

  for(int i=StartFile;i<=FinFile;++i){
    sprintf(filename,"%s/CTF/TkAlCaRecoSkimming.ALCARECOTkAlMinBias.ALCARECOTkAlMinBias_cfg.%d_TrackStats.root",basepath,i);
    // if(!CheckFileExistence(filename)){
    //  cout<<"Added to the chain "<<i-1<<" files"<<endl;
    //  break;
    // }
    if(CheckFileExistence(filename)){ch->Add(filename);addedfiles++;}
    if( (i%50==0) || (i==FinFile))cout<<"File: "<<i<<" "<<flush;
  }
  cout<<"\nAdded to the chain "<<addedfiles<<" files"<<endl;

  //create histograms with the TTree:Draw function
  ch->Draw("Chi2n>>hchi2n(600,0.0,12.0)","Ntracks>0","goff");
  TH1F *hchi2n=(TH1F*)gDirectory->Get("hchi2n");
  ch->Draw("Eta>>heta(60,-3.0,3.0)","Ntracks>0","goff");
  TH1F *heta=(TH1F*)gDirectory->Get("heta");
  ch->Draw("Phi>>hphi(628,-3.14,3.14)","Ntracks>0","goff");
  TH1F *hphi=(TH1F*)gDirectory->Get("hphi");
  ch->Draw("Nhits[][0]>>hnhits(40,0,40)","Ntracks>0","goff");
  TH1F *hnhits=(TH1F*)gDirectory->Get("hnhits");
  hnhits->SetTitle("Total # hits for CTFSkimmed");
  ch->Draw("P>>hmom(1005,0,1005.0)","Ntracks>0&&P<1000.0","goff");
  TH1F *hmom=(TH1F*)gDirectory->Get("hmom");
  ch->Draw("Ntracks>>hntrks(12,0,12.0)","","goff");
  TH1F *hntrks=(TH1F*)gDirectory->Get("hntrks");
  hchi2n->SetTitle("Normalised #chi^{2} (ALL TRKS)");
  hmom->SetTitle("Momentum distribution (ALL TRKS && P<1000 GeV)");
  heta->SetTitle("#eta distribution (ALL TRKS)");
  hphi->SetTitle("#phi distribution (ALL TRKS)");
  hntrks->SetTitle("Number of tracks in event passing cuts");

  ch->Draw("Chi2n>>hchi2nPXB(600,0.0,12.0)","Ntracks>0&&(Nhits[][1]>0||Nhits[][2]>0)","goff");
  TH1F *hchi2nPXB=(TH1F*)gDirectory->Get("hchi2nPXB");
  ch->Draw("Eta>>hetaPXB(60,-3.0,3.0)","Ntracks>0&&Nhits[][1]>0","goff");
  TH1F *hetaPXB=(TH1F*)gDirectory->Get("hetaPXB");
  ch->Draw("Phi>>hphiPXB(628,-3.14,3.14)","Ntracks>0&&Nhits[][1]>0","goff");
  TH1F *hphiPXB=(TH1F*)gDirectory->Get("hphiPXB");
  ch->Draw("Nhits[][1]>>hnhitsPXB(8,0,8)","Ntracks>0&&(Nhits[][1]>0||Nhits[][2]>0)","goff");
  TH1F *hnhitsPXB=(TH1F*)gDirectory->Get("hnhitsPXB");
  hnhitsPXB->SetTitle("Total # hits in PXB (TRKS >=1 PXB hits)");
  ch->Draw("Nhits[][0]>>hnhitstotPXB(40,0,40)","Ntracks>0&&(Nhits[][1]>0||Nhits[][2]>0)","goff");
  TH1F *hnhitstotPXB=(TH1F*)gDirectory->Get("hnhitstotPXB");
  hnhitstotPXB->SetTitle("Total # hits (TRKS >=1 PXB hits)");
  ch->Draw("P>>hmomPXB(1005,0,1005.0)","Ntracks>0&&P<1000.0&&(Nhits[][1]>0||Nhits[][2]>0)","goff");
  TH1F *hmomPXB=(TH1F*)gDirectory->Get("hmomPXB");
  hchi2nPXB->SetTitle("Normalised #chi^{2} (TRKS >=1 PXB hits)");
  hmomPXB->SetTitle("Momentum distribution (TRKS >=1 PXB hits && P<1000 GeV)");
  hetaPXB->SetTitle("#eta distribution (TRKS >=1 PXB hits)");
  hphiPXB->SetTitle("#phi distribution (TRKS >=1 PXB hits)");


  //plot them
  TCanvas *c1=new TCanvas("cantrkstats1","Track Stats 1",1000,1200);
  c1->Divide(2,3);
  c1->cd(1);
  hchi2n->Draw();
  c1->cd(2);
  heta->Draw();
  c1->cd(3);
  hphi->Draw();
  c1->cd(4);
  gPad->SetLogy();
  hmom->Draw();
  c1->cd(5);
  hnhits->Draw();
  c1->cd(6);
  hntrks->Draw();


  TCanvas *c2=new TCanvas("cantrkstats2","Track Stats PXB",1000,1200);
  c2->Divide(2,3);
  c2->cd(1);
  hchi2nPXB->Draw();
  c2->cd(2);
  hetaPXB->Draw();
  c2->cd(3);
  hphiPXB->Draw();
  c2->cd(4);
  gPad->SetLogy();
  hmomPXB->Draw();
  c2->cd(6);
  hnhitsPXB->Draw();
  c2->cd(5);
  hnhitstotPXB->Draw();

  //save png files
  c1->SaveAs("./ALCARECOTkAlMinBias_CTFSkimmed_TrackStats_ALLTRKS.png");
  c2->SaveAs("./ALCARECOTkAlMinBias_CTFSkimmed_TrackStats_PIXTRKS.png");
  delete c1;
  delete c2;
  

  cout<<"Total Events: "<<ch->Draw("Ntracks","","goff")<<endl;
  cout<<"Total tracks: "<<ch->Draw("Ntracks","Ntracks>0","goff")<<endl;
  cout<<"Tracks with P<1000.0 GeV: "<<ch->Draw("Ntracks","Ntracks>0&&P<1000.0","goff") <<endl;
  ch->Draw("Nhits[][0]:P>>hnhvsp(1000,0.0,1000.0,40,0,40)","Ntracks>0&&P<1000.0","goff");
  TH2F *hnhvsp=(TH2F*)gDirectory->Get("hnhvsp");
  hnhvsp->SetMarkerStyle(7);
  hnhvsp->SetXTitle("P (GeV)");
  hnhvsp->SetYTitle("# TOT hits");
  hnhvsp->SetTitle("Correlation momentum vs #hits of CTF tracks");
  TCanvas *can_nhvsp=new TCanvas("cnhvsp","cnhvsp",900,900);
  can_nhvsp->cd();
  //  cout<<"start draw"<<endl;
  hnhvsp->Draw("colz");
  //cout<<"start save"<<endl;
  can_nhvsp->SaveAs("./ALCARECOTkAlMinBias_CTFSkimmed_NhitsVsMom.ps");
}

bool CheckFileExistence(char *filename){

  // cout<<"Checking file "<<filename<<endl;
  bool flag = true;
  
  ifstream fin(filename,ios::in);
  if(fin.fail())flag=false;
  fin.close();
  return flag;
}
