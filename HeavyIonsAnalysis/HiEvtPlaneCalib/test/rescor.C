#include "TTree.h"
#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TString.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLatex.h"
#include "TMath.h"
#include "TPaveText.h"
#include <iostream>
#include <iomanip>

using namespace std;

#include "/home/sanders/CMSSW_5_3_20_dev/src/RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneList.h"
using namespace hi;

TFile * tf;
TTree * tr;

void ResCor(Double_t mincent, Double_t maxcent, Double_t delcent, Double_t minvtx, Double_t maxvtx ){
  FILE * fout[NumEPNames];
  for(int i = 0; i<NumEPNames; i++) {
    fout[i] = fopen(Form("RescorTables/%s_%04.1f.dat",EPNames[i].data(),delcent),"w");
  }
  tf = new TFile("data/rpflat_combined.root");
  tr = (TTree *) tf->Get("hiEvtPlaneFlatCalib/tree");
  int nbins = (int) ( (maxcent-mincent)/delcent+0.1 );
  Double_t full[50];
  Double_t full_orig[50];
  Double_t full_offset[50];
  Double_t Cent;
  Double_t Vtx;
  tr->SetBranchAddress("EP",full);
  tr->SetBranchAddress("EP_orig",full_orig);
  tr->SetBranchAddress("EP_offset",full_offset);
  tr->SetBranchAddress("Cent",&Cent);
  tr->SetBranchAddress("Vtx",&Vtx);
  double cnt[200][50];
  double cora[200][50];
  double corb[200][50];
  double corc[200][50];
  double cora2[200][50];
  double corb2[200][50];
  double corc2[200][50];
  double siga;
  double sigb;
  double sigc;
  for(int i = 0; i<200; i++) {
    for(int j = 0; j<50; j++) {
      cnt[i][j]=0; cora[i][j]=0; corb[i][j]=0; corc[i][j]=0; cora2[i][j]=0; corb2[i][j]=0; corc2[i][j]=0;
    }
  }
  for(int ievent = 0; ievent<tr->GetEntries(); ievent++) {
    tr->GetEntry(ievent);
    if(Cent < mincent) continue;
    if(Cent > maxcent) continue;
    if(Vtx < minvtx) continue;
    if(Vtx > maxvtx) continue;
    for(int i = 0; i<NumEPNames; i++) {
    double order = EPOrder[i];
    double ang[50];
    double ang1[50];
    double ang2[50];
    ang[i] = full[i];
    ang1[i] = full[RCMate1[i]];
    ang2[i] = full[RCMate2[i]];
    int bin = (Cent-mincent)/delcent;
    if(ResCalcType[i][0]=='3') {
      if(ang[i]>-5&&ang1[i]>-5&&ang2[i]>-5) {
	cora[bin][i]+=TMath::Cos( order*(ang[i] - ang1[i]) );
	cora2[bin][i]+=pow(TMath::Cos( order*(ang[i] - ang1[i]) ),2);
	corb[bin][i]+=TMath::Cos( order*(ang[i] - ang2[i]) );
	corb2[bin][i]+=pow(TMath::Cos( order*(ang[i] - ang2[i]) ),2);
	corc[bin][i]+=TMath::Cos( order*(ang2[i] - ang1[i]) );
	corc2[bin][i]+=pow(TMath::Cos( order*(ang2[i] - ang1[i]) ),2);
	++cnt[bin][i];
      }
    } else {
      if(ang[i]>-5 && ang1[i]>-5) {
	cora[bin][i]+=TMath::Cos( order*(ang[i] - ang1[i]) );
	cora2[bin][i]+=pow(TMath::Cos( order*(ang[i] - ang1[i]) ), 2);
	++cnt[bin][i];
      }
    }
    }
  }
  for(int i = 0; i<nbins; i++) {
    for(int j = 0; j<NumEPNames; j++) {
      if(cnt[i][j]<=0) continue; 
      cora[i][j]/=cnt[i][j];
      corb[i][j]/=cnt[i][j];
      corc[i][j]/=cnt[i][j];
      cora2[i][j]/=cnt[i][j];
      corb2[i][j]/=cnt[i][j];
      corc2[i][j]/=cnt[i][j];
      siga = sqrt(cora2[i][j] - pow(cora[i][j],2))/sqrt(cnt[i][j]);
      if(ResCalcType[j][0]=='3') {
	sigb = sqrt(corb2[i][j] - pow(corb[i][j],2))/sqrt(cnt[i][j]);
	sigc = sqrt(corc2[i][j] - pow(corc[i][j],2))/sqrt(cnt[i][j]);
      }
      double resc = 0;
      double err = 0;
      if(ResCalcType[j][0]=='3') {
	resc = cora[i][j] * corb[i][j]/corc[i][j];
	err = resc*sqrt(pow(siga/cora[i][j],2)+pow(sigb/corb[i][j],2)+pow(sigc/corc[i][j],2));
	if(resc>0) {
	  resc = TMath::Sqrt(resc);
	  err = 0.5*err/resc;
	} else {
	  resc = -resc;
	  resc = TMath::Sqrt(resc);
	  err = -0.5*err/resc;
	}
	
      } else {
	resc = cora[i][j] ;
	if(resc>0) {
	  resc = sqrt(resc);
	  err = 0.5*siga/resc;
	} else {
	  resc = -resc;
	  resc = sqrt(resc);
	  err = 0.5*fabs(siga)/resc;
	}
      }
      double cmin = mincent+i*delcent;
      double cmax = cmin+delcent;
      fprintf(fout[j],"%5.1f\t%5.1f\t%7.5f\t%7.5f\n",cmin,cmax,resc,err);

    }
  }
  for(int i = 0; i<NumEPNames; i++) fclose(fout[i]);
  return;
}

void rescor(){
  ResCor(0,100,1,-15,15);
  ResCor(0,100,2,-15,15);
  ResCor(0,100,5,-15,15);
  ResCor(0,100,10,-15,15);
  ResCor(0,100,20,-15,15);
  ResCor(0,100,25,-15,15);
  ResCor(0,90,30,-15,15);
  ResCor(0,80,40,-15,15);
}
