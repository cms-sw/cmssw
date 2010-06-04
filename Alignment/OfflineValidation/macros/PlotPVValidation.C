#include <string>
#include <sstream>
#include <vector>
#include <Riostream.h>
#include <iomanip>
#include "TFile.h"
#include "TPaveStats.h"
#include "TList.h"
#include "TNtuple.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "THStack.h"
#include "TStyle.h"
#include "TLegendEntry.h"
#include "TCut.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "TMath.h"
#include "TVectorD.h"
#include <string>
#include <sstream>
#include <vector>
#include <Riostream.h>
#include <iomanip>
#include "TFile.h"
#include "TPaveStats.h"
#include "THStack.h"
#include "TCut.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TLegend.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "TString.h"
#include "TMath.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

TList *FileList;
TList *LabelList;
TFile *Target;

void PlotPVValidation(TString namesandlabels,Int_t nOfFiles, TCut theCut);
void fitResiduals(TH1 *hist);

void PlotPVValidation(TString namesandlabels,Int_t nOfFiles, TCut theCut){

  TH1::StatOverflows(kTRUE);
  //gStyle->SetOptTitle(0);
  gStyle->SetOptStat("emr");
  gStyle->SetPadTopMargin(0.11);
  gStyle->SetPadBottomMargin(0.12);
  gStyle->SetPadLeftMargin(0.13);
  gStyle->SetPadRightMargin(0.02);
  gStyle->SetPadBorderMode(0);
  gStyle->SetTitleFillColor(10);
  gStyle->SetTitleFont(42);
  gStyle->SetTitleColor(1);
  gStyle->SetTitleTextColor(1);
  gStyle->SetTitleFontSize(0.06);
  gStyle->SetTitleBorderSize(0);
  gStyle->SetStatColor(kWhite);
  gStyle->SetStatFont(42);
  gStyle->SetStatFontSize(0.05);///---> gStyle->SetStatFontSize(0.025);
  gStyle->SetStatTextColor(1);
  gStyle->SetStatFormat("6.4g");
  gStyle->SetStatBorderSize(1);
  gStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  gStyle->SetPadTickY(1);
  gStyle->SetPadBorderMode(0);
  gStyle->SetOptFit(1);

  using namespace std;
  using namespace TMath;

  FileList = new TList();
  LabelList = new TList();
  
  TObjArray *nameandlabelpairs = namesandlabels.Tokenize(",");
  for (Int_t i = 0; i < nameandlabelpairs->GetEntries(); ++i) {
    TObjArray *aFileLegPair = TString(nameandlabelpairs->At(i)->GetName()).Tokenize("=");
    
    if(aFileLegPair->GetEntries() == 2) {
      FileList->Add( TFile::Open(aFileLegPair->At(0)->GetName())  );  // 2
      LabelList->Add( aFileLegPair->At(1) );
    }
    else {
      std::cout << "Please give file name and legend entry in the following form:\n" 
		<< " filename1=legendentry1,filename2=legendentry2\n";
      
    }    
  }
  
  Int_t NOfFiles =  FileList->GetSize();
  if(NOfFiles!=nOfFiles){
    cout<<"&MSG-e: NOfFiles = "<<nOfFiles<<" != "<<NOfFiles<<" given as input"<<endl;
    return;
  }  
  
  TTree *trees[nOfFiles]; 
  TString LegLabels[nOfFiles];  

  for(Int_t j=0; j < nOfFiles; j++) {
    
    // Retrieve files

    TFile *fin = (TFile*)FileList->At(j);    
    trees[j] = (TTree*)fin->Get("tree");

    // Retrieve labels

    TObjString* legend = (TObjString*)LabelList->At(j);
    LegLabels[j] = legend->String();
    cout<<"LegLabels["<<j<<"]"<<LegLabels[j]<<endl;
    
  }

  TString label = LegLabels[nOfFiles-1];
   

  //#################### Canvases #####################

  TCanvas *c1 = new TCanvas("c1","c1",800,550);
  c1->SetFillColor(10);  
  
  TCanvas *c2 = new TCanvas("c2","c2",1460,600);
  c2->SetFillColor(10);  
  c2->Divide(6,2);
  
  TCanvas *c3 = new TCanvas("c3","c3",1460,600);
  c3->SetFillColor(10);  
  c3->Divide(6,2);

  TCanvas *c2Eta = new TCanvas("c2Eta","c2Eta",1460,600);
  c2Eta->SetFillColor(10);  
  c2Eta->Divide(6,2);
  
  TCanvas *c3Eta = new TCanvas("c3Eta","c3Eta",1460,600);
  c3Eta->SetFillColor(10);  
  c3Eta->Divide(6,2);
   
  TCanvas *c4 = new TCanvas("c4","c4",1600,550);
  c4->SetFillColor(10);  
  c4->Divide(2,1);
  
  TCanvas *c5 = new TCanvas("c5","c5",1600,550);
  c5->SetFillColor(10);  
  c5->Divide(2,1);

  TCanvas *c4Eta = new TCanvas("c4Eta","c4Eta",1600,550);
  c4Eta->SetFillColor(10);  
  c4Eta->Divide(2,1);
  
  TCanvas *c5Eta = new TCanvas("c5Eta","c5Eta",1600,550);
  c5Eta->SetFillColor(10);  
  c5Eta->Divide(2,1);
               
  TLegend *lego = new TLegend(0.15,0.75,0.39,0.89);
  lego->SetFillColor(10);

  Int_t nplots = 12; 
  
  Int_t colors[8]={1,2,4,6,5,3,8,9};
  Int_t styles[8]={20,4,21,22,25,28,4,29}; 


  TH1F *histosdxy[nOfFiles][nplots];
  TH1F *histosdz[nOfFiles][nplots];

  TH1F *histosEtadxy[nOfFiles][nplots];
  TH1F *histosEtadz[nOfFiles][nplots];

  //############################
  // Canvas of bin residuals
  //############################
  
  for(Int_t i=0; i<nplots; i++){
    for(Int_t j=0; j < nOfFiles; j++){

      float phiF=(-3.141+i*0.5235)*(180/3.141);
      float phiL=(-3.141+(i+1)*0.5235)*(180/3.141);

      float etaF=-2.5+i*0.417;
      float etaL=-2.5+(i+1)*0.417;
      
      //###################### vs phi #####################

      char histotitledxy[129];
      sprintf(histotitledxy,"%.2f #circ<#phi<%.2f #circ;d_{xy} [#mum];tracks",phiF,phiL);
      char treeleg1[129];
      sprintf(treeleg1,"histodxyfile%i_plot%i",j,i);
      histosdxy[j][i] = new TH1F(treeleg1,histotitledxy,100,-500,500);
      histosdxy[j][i]->SetLineColor(colors[j]);
      //histosdxy[j][i]->SetTitleSize(0.09); 
      histosdxy[j][i]->GetYaxis()->SetTitleOffset(1.2);
      histosdxy[j][i]->GetXaxis()->SetTitleOffset(0.9);
      histosdxy[j][i]->GetYaxis()->SetTitleSize(0.08);
      histosdxy[j][i]->GetXaxis()->SetTitleSize(0.07);
      histosdxy[j][i]->GetXaxis()->SetTitleFont(42); 
      histosdxy[j][i]->GetYaxis()->SetTitleFont(42);  
      histosdxy[j][i]->GetXaxis()->SetLabelSize(0.05); 
      histosdxy[j][i]->GetYaxis()->SetLabelSize(0.05);
      histosdxy[j][i]->GetXaxis()->SetLabelFont(42);
      histosdxy[j][i]->GetYaxis()->SetLabelFont(42);

      char histotitledz[129];
      sprintf(histotitledz,"%.2f #circ<#phi<%.2f #circ;d_{z} [#mum];tracks",phiF,phiL);
      char treeleg2[129];
      sprintf(treeleg2,"histodzfile%i_plot%i",j,i);
      histosdz[j][i] = new TH1F(treeleg2,histotitledz,100,-500,500);
      histosdz[j][i]->SetLineColor(colors[j]);
      //histosdz[j][i]->SetTitleSize(0.09); 
      histosdz[j][i]->GetYaxis()->SetTitleOffset(1.2);
      histosdz[j][i]->GetXaxis()->SetTitleOffset(0.9);
      histosdz[j][i]->GetYaxis()->SetTitleSize(0.08);
      histosdz[j][i]->GetXaxis()->SetTitleSize(0.07);
      histosdz[j][i]->GetXaxis()->SetTitleFont(42); 
      histosdz[j][i]->GetYaxis()->SetTitleFont(42);  
      histosdz[j][i]->GetXaxis()->SetLabelSize(0.05); 
      histosdz[j][i]->GetYaxis()->SetLabelSize(0.05);
      histosdz[j][i]->GetXaxis()->SetLabelFont(42);
      histosdz[j][i]->GetYaxis()->SetLabelFont(42);

       //###################### vs eta ##################### 

      char histotitledxyeta[129];
      sprintf(histotitledxyeta,"%.2f <#eta<%.2f ;d_{xy} [#mum];tracks",etaF,etaL);
      char treeleg1eta[129];
      sprintf(treeleg1eta,"histodxyEtafile%i_plot%i",j,i);
      histosEtadxy[j][i] = new TH1F(treeleg1eta,histotitledxyeta,100,-500,500);
      histosEtadxy[j][i]->SetLineColor(colors[j]);
      //histosdxy[j][i]->SetTitleSize(0.09); 
      histosEtadxy[j][i]->GetYaxis()->SetTitleOffset(1.2);
      histosEtadxy[j][i]->GetXaxis()->SetTitleOffset(0.9);
      histosEtadxy[j][i]->GetYaxis()->SetTitleSize(0.08);
      histosEtadxy[j][i]->GetXaxis()->SetTitleSize(0.07);
      histosEtadxy[j][i]->GetXaxis()->SetTitleFont(42); 
      histosEtadxy[j][i]->GetYaxis()->SetTitleFont(42);  
      histosEtadxy[j][i]->GetXaxis()->SetLabelSize(0.05); 
      histosEtadxy[j][i]->GetYaxis()->SetLabelSize(0.05);
      histosEtadxy[j][i]->GetXaxis()->SetLabelFont(42);
      histosEtadxy[j][i]->GetYaxis()->SetLabelFont(42);
      
      char histotitledzeta[129];
      sprintf(histotitledzeta,"%.2f <#eta<%.2f ;d_{z} [#mum];tracks",etaF,etaL);
      char treeleg2eta[129];
      sprintf(treeleg2eta,"histodzEtafile%i_plot%i",j,i);
      histosEtadz[j][i] = new TH1F(treeleg2eta,histotitledzeta,100,-500,500);
      histosEtadz[j][i]->SetLineColor(colors[j]);
      //histosdz[j][i]->SetTitleSize(0.09); 
      histosEtadz[j][i]->GetYaxis()->SetTitleOffset(1.2);
      histosEtadz[j][i]->GetXaxis()->SetTitleOffset(0.9);
      histosEtadz[j][i]->GetYaxis()->SetTitleSize(0.08);
      histosEtadz[j][i]->GetXaxis()->SetTitleSize(0.07);
      histosEtadz[j][i]->GetXaxis()->SetTitleFont(42); 
      histosEtadz[j][i]->GetYaxis()->SetTitleFont(42);  
      histosEtadz[j][i]->GetXaxis()->SetLabelSize(0.05); 
      histosEtadz[j][i]->GetYaxis()->SetLabelSize(0.05);
      histosEtadz[j][i]->GetXaxis()->SetLabelFont(42);
      histosEtadz[j][i]->GetYaxis()->SetLabelFont(42);
      
    }
  }
  
  cout<<"Checkpoint 1: before anything"<<endl;

  //################ Phi #####################

  TObject *statObjsdxy[nOfFiles][nplots];
  TPaveStats *statsdxy[nOfFiles][nplots];
  TObject *statObjsdz[nOfFiles][nplots];
  TPaveStats *statsdz[nOfFiles][nplots];
  
  TH1F *histoMeansdxy[nOfFiles];  
  TH1F *histoMeansdz[nOfFiles];     
  TH1F *histoSigmasdxy[nOfFiles];   
  TH1F *histoSigmasdz[nOfFiles];    
  
  TF1 *tmpsdxy[nOfFiles][nplots];
  TF1 *tmpsdz[nOfFiles][nplots];
  
  Double_t meansdxy[nOfFiles][nplots];
  Double_t meansdxyError[nOfFiles][nplots];
  //Double_t mediansdxy[nOfFiles][nplots];
  Double_t sigmasdxy[nOfFiles][nplots];
  Double_t sigmasdxyError[nOfFiles][nplots];

  Double_t meansdz[nOfFiles][nplots];
  Double_t meansdzError[nOfFiles][nplots];
  //Double_t mediansdz[nOfFiles][nplots];
  Double_t sigmasdz[nOfFiles][nplots];
  Double_t sigmasdzError[nOfFiles][nplots];

  //################ Eta #####################

  TObject *statObjsdxyEta[nOfFiles][nplots];
  TPaveStats *statsdxyEta[nOfFiles][nplots];
  TObject *statObjsdzEta[nOfFiles][nplots];
  TPaveStats *statsdzEta[nOfFiles][nplots]; 

  TH1F *histoMeansdxyEta[nOfFiles];  
  TH1F *histoMeansdzEta[nOfFiles];     
  TH1F *histoSigmasdxyEta[nOfFiles];   
  TH1F *histoSigmasdzEta[nOfFiles];    
  
  TF1 *tmpsdxyEta[nOfFiles][nplots];
  TF1 *tmpsdzEta[nOfFiles][nplots];
  
  Double_t meansdxyEta[nOfFiles][nplots];
  Double_t meansdxyEtaError[nOfFiles][nplots];
  //Double_t mediansdxy[nOfFiles][nplots];
  Double_t sigmasdxyEta[nOfFiles][nplots];
  Double_t sigmasdxyEtaError[nOfFiles][nplots];

  Double_t meansdzEta[nOfFiles][nplots];
  Double_t meansdzEtaError[nOfFiles][nplots];
  //Double_t mediansdz[nOfFiles][nplots];
  Double_t sigmasdzEta[nOfFiles][nplots];
  Double_t sigmasdzEtaError[nOfFiles][nplots];

   for(Int_t j=0; j < nOfFiles; j++) { 

     //################ phi #########################

    char meanslegdxy[129];
    sprintf(meanslegdxy,"histomeansdxy_file%i",j);
    histoMeansdxy[j]  = new TH1F(meanslegdxy,"<d_{xy}> vs #phi sector",12,-0.5,11.5);
    histoMeansdxy[j]->SetLineColor(colors[j]);
    histoMeansdxy[j]->SetStats(kFALSE);
    histoMeansdxy[j]->GetYaxis()->SetTitleOffset(1.2);
    histoMeansdxy[j]->GetXaxis()->SetTitleOffset(0.9);
    histoMeansdxy[j]->GetYaxis()->SetTitleSize(0.08);
    histoMeansdxy[j]->GetXaxis()->SetTitleSize(0.07);
    histoMeansdxy[j]->GetXaxis()->SetTitleFont(42); 
    histoMeansdxy[j]->GetYaxis()->SetTitleFont(42);  
    histoMeansdxy[j]->GetXaxis()->SetLabelSize(0.05); 
    histoMeansdxy[j]->GetYaxis()->SetLabelSize(0.05);
    histoMeansdxy[j]->GetXaxis()->SetLabelFont(42);
    histoMeansdxy[j]->GetYaxis()->SetLabelFont(42);
    histoMeansdxy[j]->GetYaxis()->SetRangeUser(-120,120);
    histoMeansdxy[j]->SetLineWidth(2);
    histoMeansdxy[j]->SetMarkerStyle(styles[j]);
    histoMeansdxy[j]->SetMarkerColor(colors[j]);
    histoMeansdxy[j]->GetXaxis()->SetTitle("#phi (sector)");
    histoMeansdxy[j]->GetYaxis()->SetTitle ("d_{xy} (#mum)");
    histoMeansdxy[j]->GetXaxis()->CenterTitle(true);
    histoMeansdxy[j]->GetYaxis()->CenterTitle(true);
    histoMeansdxy[j]->GetXaxis()->SetTitleSize(0.06);
    histoMeansdxy[j]->GetYaxis()->SetTitleSize(0.06);
    histoMeansdxy[j]->GetXaxis()->SetTitleOffset(1.0);
    histoMeansdxy[j]->GetYaxis()->SetTitleOffset(1.0);
    histoMeansdxy[j]->GetYaxis()->SetLabelSize(.06);
    histoMeansdxy[j]->GetXaxis()->SetLabelSize(.06);
   
    char meanslegdz[129];
    sprintf(meanslegdz,"histomeansdz_file%i",j);
    histoMeansdz[j]   = new TH1F(meanslegdz,"<d_{z}> vs #phi sector",12,-0.5,11.5); 
    histoMeansdz[j]->SetLineColor(colors[j]);
    histoMeansdz[j]->SetStats(kFALSE);
    histoMeansdz[j]->GetYaxis()->SetTitleOffset(1.2);
    histoMeansdz[j]->GetXaxis()->SetTitleOffset(0.9);
    histoMeansdz[j]->GetYaxis()->SetTitleSize(0.08);
    histoMeansdz[j]->GetXaxis()->SetTitleSize(0.07);
    histoMeansdz[j]->GetXaxis()->SetTitleFont(42); 
    histoMeansdz[j]->GetYaxis()->SetTitleFont(42);  
    histoMeansdz[j]->GetXaxis()->SetLabelSize(0.05); 
    histoMeansdz[j]->GetYaxis()->SetLabelSize(0.05);
    histoMeansdz[j]->GetXaxis()->SetLabelFont(42);
    histoMeansdz[j]->GetYaxis()->SetLabelFont(42);
    histoMeansdz[j]->GetYaxis()->SetRangeUser(-90,90);
    histoMeansdz[j]->SetLineWidth(2);
    histoMeansdz[j]->SetMarkerStyle(styles[j]);
    histoMeansdz[j]->SetMarkerColor(colors[j]);
    histoMeansdz[j]->GetXaxis()->SetTitle("#phi (sector)");
    histoMeansdz[j]->GetYaxis()->SetTitle ("<d_{z}> (#mum)");
    histoMeansdz[j]->GetXaxis()->CenterTitle(true);
    histoMeansdz[j]->GetYaxis()->CenterTitle(true);
    histoMeansdz[j]->GetXaxis()->SetTitleSize(0.06);
    histoMeansdz[j]->GetYaxis()->SetTitleSize(0.06);
    histoMeansdz[j]->GetXaxis()->SetTitleOffset(1.0);
    histoMeansdz[j]->GetYaxis()->SetTitleOffset(1.0);
    histoMeansdz[j]->GetYaxis()->SetLabelSize(.06);
    histoMeansdz[j]->GetXaxis()->SetLabelSize(.06);
    histoMeansdz[j]->Draw("e1sames");

    char sigmaslegdxy[129];
    sprintf(sigmaslegdxy,"histosigmasdxy_file%i",j);
    histoSigmasdxy[j] = new TH1F(sigmaslegdxy,"#sigma_{d_{xy}} vs #phi sector",12,-0.5,11.5);
    histoSigmasdxy[j]->SetLineColor(colors[j]);
    histoSigmasdxy[j]->SetStats(kFALSE);
    histoSigmasdxy[j]->GetYaxis()->SetTitleOffset(1.2);
    histoSigmasdxy[j]->GetXaxis()->SetTitleOffset(0.9);
    histoSigmasdxy[j]->GetYaxis()->SetTitleSize(0.08);
    histoSigmasdxy[j]->GetXaxis()->SetTitleSize(0.07);
    histoSigmasdxy[j]->GetXaxis()->SetTitleFont(42); 
    histoSigmasdxy[j]->GetYaxis()->SetTitleFont(42);  
    histoSigmasdxy[j]->GetXaxis()->SetLabelSize(0.05); 
    histoSigmasdxy[j]->GetYaxis()->SetLabelSize(0.05);
    histoSigmasdxy[j]->GetXaxis()->SetLabelFont(42);
    histoSigmasdxy[j]->GetYaxis()->SetLabelFont(42);
    histoSigmasdxy[j]->GetYaxis()->SetRangeUser(150,310);
    histoSigmasdxy[j]->SetLineWidth(2);
    histoSigmasdxy[j]->SetMarkerStyle(styles[j]);
    histoSigmasdxy[j]->SetMarkerColor(colors[j]);
    histoSigmasdxy[j]->GetXaxis()->SetTitle("#phi (sector)");
    histoSigmasdxy[j]->GetYaxis()->SetTitle ("#sigma_{d_{xy}} (#mum)");
    histoSigmasdxy[j]->GetXaxis()->CenterTitle(true);
    histoSigmasdxy[j]->GetYaxis()->CenterTitle(true);
    histoSigmasdxy[j]->GetXaxis()->SetTitleSize(0.06);
    histoSigmasdxy[j]->GetYaxis()->SetTitleSize(0.06);
    histoSigmasdxy[j]->GetXaxis()->SetTitleOffset(1.0);
    histoSigmasdxy[j]->GetYaxis()->SetTitleOffset(1.2);
    histoSigmasdxy[j]->GetYaxis()->SetLabelSize(.06);
    histoSigmasdxy[j]->GetXaxis()->SetLabelSize(.06);

    char sigmaslegdz[129];
    sprintf(sigmaslegdz,"histosigmasdz_file%i",j);
    histoSigmasdz[j]  = new TH1F(sigmaslegdz,"#sigma_{d_{z}} vs #phi sector",12,-0.5,11.5);
    histoSigmasdz[j]->SetLineColor(colors[j]);
    histoSigmasdz[j]->SetStats(kFALSE);
    histoSigmasdz[j]->GetYaxis()->SetTitleOffset(1.2);
    histoSigmasdz[j]->GetXaxis()->SetTitleOffset(0.9);
    histoSigmasdz[j]->GetYaxis()->SetTitleSize(0.08);
    histoSigmasdz[j]->GetXaxis()->SetTitleSize(0.07);
    histoSigmasdz[j]->GetXaxis()->SetTitleFont(42); 
    histoSigmasdz[j]->GetYaxis()->SetTitleFont(42);  
    histoSigmasdz[j]->GetXaxis()->SetLabelSize(0.05); 
    histoSigmasdz[j]->GetYaxis()->SetLabelSize(0.05);
    histoSigmasdz[j]->GetXaxis()->SetLabelFont(42);
    histoSigmasdz[j]->GetYaxis()->SetLabelFont(42);
    histoSigmasdz[j]->GetYaxis()->SetRangeUser(200,510);
    histoSigmasdz[j]->SetLineWidth(2);
    histoSigmasdz[j]->SetMarkerStyle(styles[j]);
    histoSigmasdz[j]->SetMarkerColor(colors[j]);
    histoSigmasdz[j]->GetXaxis()->SetTitle("#phi (sector)");
    histoSigmasdz[j]->GetYaxis()->SetTitle ("#sigma_{d_{z}} (#mum)");
    histoSigmasdz[j]->GetXaxis()->CenterTitle(true);
    histoSigmasdz[j]->GetYaxis()->CenterTitle(true);
    histoSigmasdz[j]->GetXaxis()->SetTitleSize(0.06);
    histoSigmasdz[j]->GetYaxis()->SetTitleSize(0.06);
    histoSigmasdz[j]->GetXaxis()->SetTitleOffset(1.0);
    histoSigmasdz[j]->GetYaxis()->SetTitleOffset(1.2);
    histoSigmasdz[j]->GetYaxis()->SetLabelSize(.06);
    histoSigmasdz[j]->GetXaxis()->SetLabelSize(.06);	   

    //################ eta #########################
  
    char meanslegdxyEta[129];
    sprintf(meanslegdxyEta,"histomeansEtadxy_file%i",j);
    histoMeansdxyEta[j]  = new TH1F(meanslegdxyEta,"<d_{xy}> vs #eta sector",12,-0.5,11.5);
    histoMeansdxyEta[j]->SetLineColor(colors[j]);
    histoMeansdxyEta[j]->SetStats(kFALSE);
    histoMeansdxyEta[j]->GetYaxis()->SetTitleOffset(1.2);
    histoMeansdxyEta[j]->GetXaxis()->SetTitleOffset(0.9);
    histoMeansdxyEta[j]->GetYaxis()->SetTitleSize(0.08);
    histoMeansdxyEta[j]->GetXaxis()->SetTitleSize(0.07);
    histoMeansdxyEta[j]->GetXaxis()->SetTitleFont(42); 
    histoMeansdxyEta[j]->GetYaxis()->SetTitleFont(42);  
    histoMeansdxyEta[j]->GetXaxis()->SetLabelSize(0.05); 
    histoMeansdxyEta[j]->GetYaxis()->SetLabelSize(0.05);
    histoMeansdxyEta[j]->GetXaxis()->SetLabelFont(42);
    histoMeansdxyEta[j]->GetYaxis()->SetLabelFont(42);
    histoMeansdxyEta[j]->GetYaxis()->SetRangeUser(-40,40);
    histoMeansdxyEta[j]->SetLineWidth(2);
    histoMeansdxyEta[j]->SetMarkerStyle(styles[j]);
    histoMeansdxyEta[j]->SetMarkerColor(colors[j]);
    histoMeansdxyEta[j]->GetXaxis()->SetTitle("#eta (sector)");
    histoMeansdxyEta[j]->GetYaxis()->SetTitle ("d_{xy} (#mum)");
    histoMeansdxyEta[j]->GetXaxis()->CenterTitle(true);
    histoMeansdxyEta[j]->GetYaxis()->CenterTitle(true);
    histoMeansdxyEta[j]->GetXaxis()->SetTitleSize(0.06);
    histoMeansdxyEta[j]->GetYaxis()->SetTitleSize(0.06);
    histoMeansdxyEta[j]->GetXaxis()->SetTitleOffset(1.0);
    histoMeansdxyEta[j]->GetYaxis()->SetTitleOffset(1.0);
    histoMeansdxyEta[j]->GetYaxis()->SetLabelSize(.06);
    histoMeansdxyEta[j]->GetXaxis()->SetLabelSize(.06);
   
    char meanslegdzEta[129];
    sprintf(meanslegdzEta,"histomeansEtadz_file%i",j);
    histoMeansdzEta[j]   = new TH1F(meanslegdzEta,"<d_{z}> vs #eta sector",12,-0.5,11.5); 
    histoMeansdzEta[j]->SetLineColor(colors[j]);
    histoMeansdzEta[j]->SetStats(kFALSE);
    histoMeansdzEta[j]->GetYaxis()->SetTitleOffset(1.2);
    histoMeansdzEta[j]->GetXaxis()->SetTitleOffset(0.9);
    histoMeansdzEta[j]->GetYaxis()->SetTitleSize(0.08);
    histoMeansdzEta[j]->GetXaxis()->SetTitleSize(0.07);
    histoMeansdzEta[j]->GetXaxis()->SetTitleFont(42); 
    histoMeansdzEta[j]->GetYaxis()->SetTitleFont(42);  
    histoMeansdzEta[j]->GetXaxis()->SetLabelSize(0.05); 
    histoMeansdzEta[j]->GetYaxis()->SetLabelSize(0.05);
    histoMeansdzEta[j]->GetXaxis()->SetLabelFont(42);
    histoMeansdzEta[j]->GetYaxis()->SetLabelFont(42);
    histoMeansdzEta[j]->GetYaxis()->SetRangeUser(-160,160);
    histoMeansdzEta[j]->SetLineWidth(2);
    histoMeansdzEta[j]->SetMarkerStyle(styles[j]);
    histoMeansdzEta[j]->SetMarkerColor(colors[j]);
    histoMeansdzEta[j]->GetXaxis()->SetTitle("#eta (sector)");
    histoMeansdzEta[j]->GetYaxis()->SetTitle ("<d_{z}> (#mum)");
    histoMeansdzEta[j]->GetXaxis()->CenterTitle(true);
    histoMeansdzEta[j]->GetYaxis()->CenterTitle(true);
    histoMeansdzEta[j]->GetXaxis()->SetTitleSize(0.06);
    histoMeansdzEta[j]->GetYaxis()->SetTitleSize(0.06);
    histoMeansdzEta[j]->GetXaxis()->SetTitleOffset(1.0);
    histoMeansdzEta[j]->GetYaxis()->SetTitleOffset(1.0);
    histoMeansdzEta[j]->GetYaxis()->SetLabelSize(.06);
    histoMeansdzEta[j]->GetXaxis()->SetLabelSize(.06);
    histoMeansdzEta[j]->Draw("e1sames");

    char sigmaslegdxyEta[129];
    sprintf(sigmaslegdxyEta,"histosigmasEtadxy_file%i",j);
    histoSigmasdxyEta[j] = new TH1F(sigmaslegdxyEta,"#sigma_{d_{xy}} vs #eta sector",12,-0.5,11.5);
    histoSigmasdxyEta[j]->SetLineColor(colors[j]);
    histoSigmasdxyEta[j]->SetStats(kFALSE);
    histoSigmasdxyEta[j]->GetYaxis()->SetTitleOffset(1.2);
    histoSigmasdxyEta[j]->GetXaxis()->SetTitleOffset(0.9);
    histoSigmasdxyEta[j]->GetYaxis()->SetTitleSize(0.08);
    histoSigmasdxyEta[j]->GetXaxis()->SetTitleSize(0.07);
    histoSigmasdxyEta[j]->GetXaxis()->SetTitleFont(42); 
    histoSigmasdxyEta[j]->GetYaxis()->SetTitleFont(42);  
    histoSigmasdxyEta[j]->GetXaxis()->SetLabelSize(0.05); 
    histoSigmasdxyEta[j]->GetYaxis()->SetLabelSize(0.05);
    histoSigmasdxyEta[j]->GetXaxis()->SetLabelFont(42);
    histoSigmasdxyEta[j]->GetYaxis()->SetLabelFont(42);
    histoSigmasdxyEta[j]->GetYaxis()->SetRangeUser(120,310);
    histoSigmasdxyEta[j]->SetLineWidth(2);
    histoSigmasdxyEta[j]->SetMarkerStyle(styles[j]);
    histoSigmasdxyEta[j]->SetMarkerColor(colors[j]);
    histoSigmasdxyEta[j]->GetXaxis()->SetTitle("#eta (sector)");
    histoSigmasdxyEta[j]->GetYaxis()->SetTitle ("#sigma_{d_{xy}} (#mum)");
    histoSigmasdxyEta[j]->GetXaxis()->CenterTitle(true);
    histoSigmasdxyEta[j]->GetYaxis()->CenterTitle(true);
    histoSigmasdxyEta[j]->GetXaxis()->SetTitleSize(0.06);
    histoSigmasdxyEta[j]->GetYaxis()->SetTitleSize(0.06);
    histoSigmasdxyEta[j]->GetXaxis()->SetTitleOffset(1.0);
    histoSigmasdxyEta[j]->GetYaxis()->SetTitleOffset(1.2);
    histoSigmasdxyEta[j]->GetYaxis()->SetLabelSize(.06);
    histoSigmasdxyEta[j]->GetXaxis()->SetLabelSize(.06);

    char sigmaslegdzEta[129];
    sprintf(sigmaslegdzEta,"histosigmasEtadz_file%i",j);
    histoSigmasdzEta[j]  = new TH1F(sigmaslegdzEta,"#sigma_{d_{z}} vs #eta sector",12,-0.5,11.5);
    histoSigmasdzEta[j]->SetLineColor(colors[j]);
    histoSigmasdzEta[j]->SetStats(kFALSE);
    histoSigmasdzEta[j]->GetYaxis()->SetTitleOffset(1.2);
    histoSigmasdzEta[j]->GetXaxis()->SetTitleOffset(0.9);
    histoSigmasdzEta[j]->GetYaxis()->SetTitleSize(0.08);
    histoSigmasdzEta[j]->GetXaxis()->SetTitleSize(0.07);
    histoSigmasdzEta[j]->GetXaxis()->SetTitleFont(42); 
    histoSigmasdzEta[j]->GetYaxis()->SetTitleFont(42);  
    histoSigmasdzEta[j]->GetXaxis()->SetLabelSize(0.05); 
    histoSigmasdzEta[j]->GetYaxis()->SetLabelSize(0.05);
    histoSigmasdzEta[j]->GetXaxis()->SetLabelFont(42);
    histoSigmasdzEta[j]->GetYaxis()->SetLabelFont(42);
    histoSigmasdzEta[j]->GetYaxis()->SetRangeUser(100,1000);
    histoSigmasdzEta[j]->SetLineWidth(2);
    histoSigmasdzEta[j]->SetMarkerStyle(styles[j]);
    histoSigmasdzEta[j]->SetMarkerColor(colors[j]);
    histoSigmasdzEta[j]->GetXaxis()->SetTitle("#eta (sector)");
    histoSigmasdzEta[j]->GetYaxis()->SetTitle ("#sigma_{d_{z}} (#mum)");
    histoSigmasdzEta[j]->GetXaxis()->CenterTitle(true);
    histoSigmasdzEta[j]->GetYaxis()->CenterTitle(true);
    histoSigmasdzEta[j]->GetXaxis()->SetTitleSize(0.06);
    histoSigmasdzEta[j]->GetYaxis()->SetTitleSize(0.06);
    histoSigmasdzEta[j]->GetXaxis()->SetTitleOffset(1.0);
    histoSigmasdzEta[j]->GetYaxis()->SetTitleOffset(1.2);
    histoSigmasdzEta[j]->GetYaxis()->SetLabelSize(.06);
    histoSigmasdzEta[j]->GetXaxis()->SetLabelSize(.06);

  }

   cout<<"Checkpoint 2: starts fitting xy residuals"<<endl;

  //######################## dxy Residuals #################################

  for(Int_t j=0; j < nOfFiles; j++) { 
    for(Int_t i=0; i<nplots; i++){

      //################### phi ########################

      char thePhiCutstring[128];
      sprintf(thePhiCutstring,"(phi>-3.141+%.1i*0.5235)&&(phi<-3.141+%.1i*0.5235)",i,i+1);
      TCut thePhiCut = thePhiCutstring;
      cout<<thePhiCutstring<<endl;
      
      char treeleg3[129];
      sprintf(treeleg3,"dxyFromMyVertex*10000>>histodxyfile%i_plot%i",j,i);
      
      c2->cd(i+1);
      if(j==0){
	trees[j]->Draw(treeleg3,theCut&&thePhiCut);  
      }
      else trees[j]->Draw(treeleg3,theCut&&thePhiCut,"sames");  

      fitResiduals(histosdxy[j][i]);   
      tmpsdxy[j][i] = (TF1*)histosdxy[j][i]->GetListOfFunctions()->FindObject("tmp");   
      if(tmpsdxy[j][i])  {
	tmpsdxy[j][i]->SetLineColor(colors[j]);
	tmpsdxy[j][i]->SetLineWidth(2); 
      }

      meansdxy[j][i]=(tmpsdxy[j][i])->GetParameter(1);
      sigmasdxy[j][i]=(tmpsdxy[j][i])->GetParameter(2);

      meansdxyError[j][i]=(tmpsdxy[j][i])->GetParError(1);
      sigmasdxyError[j][i]=(tmpsdxy[j][i])->GetParError(2); 
      
      histoMeansdxy[j]->SetBinContent(i+1,meansdxy[j][i]); 
      histoMeansdxy[j]->SetBinError(i+1,meansdxyError[j][i]); 
      histoSigmasdxy[j]->SetBinContent(i+1,sigmasdxy[j][i]);
      histoSigmasdxy[j]->SetBinError(i+1,sigmasdxyError[j][i]); 
      
      c2->Draw("sames");   
      c2->cd(i+1);

      statObjsdxy[j][i] = histosdxy[j][i]->GetListOfFunctions()->FindObject("stats");
      statsdxy[j][i] = static_cast<TPaveStats*>(statObjsdxy[j][i]);
      statsdxy[j][i]->SetLineColor(colors[j]);
      statsdxy[j][i]->SetTextColor(colors[j]);
      statsdxy[j][i]->SetShadowColor(10);
      statsdxy[j][i]->SetFillColor(10);
      statsdxy[j][i]->SetX1NDC(0.66);
      statsdxy[j][i]->SetY1NDC(0.80-j*0.18);
      statsdxy[j][i]->SetX2NDC(0.99);
      statsdxy[j][i]->SetY2NDC(0.99-j*0.19);
      statsdxy[j][i]->Draw("same");   

      //################### eta ########################
      
      char theEtaCutstring[128];
      sprintf(theEtaCutstring,"(eta>-2.5+%.1i*0.417)&&(eta<-2.5+%.1i*0.417)",i,i+1);
      TCut theEtaCut = theEtaCutstring;
      cout<<theEtaCutstring<<endl;
      
      char treeleg3Eta[129];
      sprintf(treeleg3Eta,"dxyFromMyVertex*10000>>histodxyEtafile%i_plot%i",j,i);
      
      char etapositionString[129];
      float etaposition = (-2.5+i*0.417)+(0.417/2);
      sprintf(etapositionString,"%.1f",etaposition);
      cout<<"etaposition: "<<etaposition<<" etapositionString: "<<etapositionString<<endl;
	
      c2Eta->cd(i+1);
      if(j==0){
	trees[j]->Draw(treeleg3Eta,theCut&&theEtaCut);  
      }
      else trees[j]->Draw(treeleg3Eta,theCut&&theEtaCut,"sames");  

      fitResiduals(histosEtadxy[j][i]);   
      tmpsdxyEta[j][i] = (TF1*)histosEtadxy[j][i]->GetListOfFunctions()->FindObject("tmp");   
      if(tmpsdxyEta[j][i])  {
	tmpsdxyEta[j][i]->SetLineColor(colors[j]);
	tmpsdxyEta[j][i]->SetLineWidth(2); 
      }

      meansdxyEta[j][i]=(tmpsdxyEta[j][i])->GetParameter(1);
      sigmasdxyEta[j][i]=(tmpsdxyEta[j][i])->GetParameter(2);
      
      meansdxyEtaError[j][i]=(tmpsdxyEta[j][i])->GetParError(1);
      sigmasdxyEtaError[j][i]=(tmpsdxyEta[j][i])->GetParError(2);

      histoMeansdxyEta[j]->SetBinContent(i+1,meansdxyEta[j][i]); 
      histoMeansdxyEta[j]->SetBinError(i+1,meansdxyEtaError[j][i]); 
      histoMeansdxyEta[j]->GetXaxis()->SetBinLabel(i+1,etapositionString);
      histoSigmasdxyEta[j]->SetBinContent(i+1,sigmasdxyEta[j][i]);
      histoSigmasdxyEta[j]->SetBinError(i+1,sigmasdxyEtaError[j][i]); 
      histoSigmasdxyEta[j]->GetXaxis()->SetBinLabel(i+1,etapositionString);
      
      c2Eta->Draw("sames");   
      c2Eta->cd(i+1);

      statObjsdxyEta[j][i] = histosEtadxy[j][i]->GetListOfFunctions()->FindObject("stats");
      statsdxyEta[j][i] = static_cast<TPaveStats*>(statObjsdxyEta[j][i]);
      statsdxyEta[j][i]->SetLineColor(colors[j]);
      statsdxyEta[j][i]->SetTextColor(colors[j]);
      statsdxyEta[j][i]->SetShadowColor(10);
      statsdxyEta[j][i]->SetFillColor(10);
      statsdxyEta[j][i]->SetX1NDC(0.66);
      statsdxyEta[j][i]->SetY1NDC(0.80-j*0.18);
      statsdxyEta[j][i]->SetX2NDC(0.99);
      statsdxyEta[j][i]->SetY2NDC(0.99-j*0.19);
      statsdxyEta[j][i]->Draw("same"); 
      
    } 
  }
   
  TString Canvas2Title ="VertexDxyResidualsPhiBin_"+label;
  TString Canvas2format=Canvas2Title+".png"; 
  c2->SaveAs(Canvas2format);
  
  TString Canvas2TitleEta ="VertexDxyResidualsEtaBin_"+label;
  TString Canvas2formatEta=Canvas2TitleEta+".png"; 
  c2Eta->SaveAs(Canvas2formatEta);

  cout<<"Checkpoint 3: starts fitting z residuals"<<endl;
 
  //######################## dz Residuals #################################

  for(Int_t j=0; j < nOfFiles; j++) { 
    for(Int_t i=0; i<nplots; i++){
    
      char thePhiCutstring[128];
      sprintf(thePhiCutstring,"(phi>-3.141+%.1i*0.5235)&&(phi<-3.141+%.1i*0.5235)",i,i+1);
      TCut thePhiCut = thePhiCutstring;
      cout<<thePhiCutstring<<endl;
      
      char treeleg4[129];
      sprintf(treeleg4,"dzFromMyVertex*10000>>histodzfile%i_plot%i",j,i);
	
      c3->cd(i+1);
      if(j==0){
	trees[j]->Draw(treeleg4,theCut&&thePhiCut);  
      }
      else trees[j]->Draw(treeleg4,theCut&&thePhiCut,"sames"); 
       
      fitResiduals(histosdz[j][i]);   
      tmpsdz[j][i] = (TF1*)histosdz[j][i]->GetListOfFunctions()->FindObject("tmp");   
      if(tmpsdz[j][i])  {
	tmpsdz[j][i]->SetLineColor(colors[j]);
	tmpsdz[j][i]->SetLineWidth(2); 
      }
        
      meansdz[j][i]=(tmpsdz[j][i])->GetParameter(1);
      sigmasdz[j][i]=(tmpsdz[j][i])->GetParameter(2);

      meansdzError[j][i]=(tmpsdz[j][i])->GetParError(1);
      sigmasdzError[j][i]=(tmpsdz[j][i])->GetParError(2);

      histoMeansdz[j]->SetBinContent(i+1,meansdz[j][i]); 
      histoMeansdz[j]->SetBinError(i+1,meansdzError[j][i]); 
      histoSigmasdz[j]->SetBinContent(i+1,sigmasdz[j][i]);
      histoSigmasdz[j]->SetBinError(i+1,sigmasdzError[j][i]);
 
      c3->Draw("sames");
      c3->cd(i+1);
	       
      statObjsdz[j][i] = histosdz[j][i]->GetListOfFunctions()->FindObject("stats");
      statsdz[j][i] = static_cast<TPaveStats*>(statObjsdz[j][i]);     
      statsdz[j][i]->SetLineColor(colors[j]);
      statsdz[j][i]->SetTextColor(colors[j]);
      statsdz[j][i]->SetFillColor(10);
      statsdz[j][i]->SetShadowColor(10);
      statsdz[j][i]->SetX1NDC(0.66);
      statsdz[j][i]->SetY1NDC(0.80-j*0.18);
      statsdz[j][i]->SetX2NDC(0.99);
      statsdz[j][i]->SetY2NDC(0.99-j*0.19);
      statsdz[j][i]->Draw("same");

      //################### eta ########################
      
      char theEtaCutstring[128];
      sprintf(theEtaCutstring,"(eta>-2.5+%.1i*0.417)&&(eta<-2.5+%.1i*0.417)",i,i+1);
      TCut theEtaCut = theEtaCutstring;
      cout<<theEtaCutstring<<endl;
      
      char treeleg3Eta[129];
      sprintf(treeleg3Eta,"dzFromMyVertex*10000>>histodzEtafile%i_plot%i",j,i);
      
      char etapositionString[129];
      float etaposition = (-2.5+i*0.417)+(0.417/2);
      sprintf(etapositionString,"%.1f",etaposition);
      //cout<<"etaposition: "<<etaposition<<" etapositionString: "<<etapositionString<<endl;

      c3Eta->cd(i+1);
      if(j==0){
	trees[j]->Draw(treeleg3Eta,theCut&&theEtaCut);  
      }
      else trees[j]->Draw(treeleg3Eta,theCut&&theEtaCut,"sames");  

      fitResiduals(histosEtadz[j][i]);   
      tmpsdzEta[j][i] = (TF1*)histosEtadz[j][i]->GetListOfFunctions()->FindObject("tmp");   
      if(tmpsdzEta[j][i])  {
	tmpsdzEta[j][i]->SetLineColor(colors[j]);
	tmpsdzEta[j][i]->SetLineWidth(2); 
      }

      meansdzEta[j][i]=(tmpsdzEta[j][i])->GetParameter(1);
      sigmasdzEta[j][i]=(tmpsdzEta[j][i])->GetParameter(2);
      
      meansdzEtaError[j][i]=(tmpsdzEta[j][i])->GetParError(1);
      sigmasdzEtaError[j][i]=(tmpsdzEta[j][i])->GetParError(2);

      histoMeansdzEta[j]->SetBinContent(i+1,meansdzEta[j][i]); 
      histoMeansdzEta[j]->SetBinError(i+1,meansdzEtaError[j][i]); 
      histoMeansdzEta[j]->GetXaxis()->SetBinLabel(i+1,etapositionString);
      histoSigmasdzEta[j]->SetBinContent(i+1,sigmasdzEta[j][i]);
      histoSigmasdzEta[j]->SetBinError(i+1,sigmasdzEtaError[j][i]);
      histoSigmasdzEta[j]->GetXaxis()->SetBinLabel(i+1,etapositionString);
      
      c3Eta->Draw("sames");   
      c3Eta->cd(i+1);

      statObjsdzEta[j][i] = histosEtadz[j][i]->GetListOfFunctions()->FindObject("stats");
      statsdzEta[j][i] = static_cast<TPaveStats*>(statObjsdzEta[j][i]);
      statsdzEta[j][i]->SetLineColor(colors[j]);
      statsdzEta[j][i]->SetTextColor(colors[j]);
      statsdzEta[j][i]->SetShadowColor(10);
      statsdzEta[j][i]->SetFillColor(10);
      statsdzEta[j][i]->SetX1NDC(0.66);
      statsdzEta[j][i]->SetY1NDC(0.80-j*0.18);
      statsdzEta[j][i]->SetX2NDC(0.99);
      statsdzEta[j][i]->SetY2NDC(0.99-j*0.19);
      statsdzEta[j][i]->Draw("same"); 
 
    }
  }
      
  TString Canvas3Title ="VertexDzResidualsPhiBin_"+label;
  TString Canvas3format=Canvas3Title+".png"; 
  c3Eta->SaveAs(Canvas3format);

  TString Canvas3TitleEta ="VertexDzResidualsEtaBin_"+label;
  TString Canvas3formatEta=Canvas3TitleEta+".png"; 
  c3Eta->SaveAs(Canvas3formatEta);

  //######################################################
  //  Histograms Means and Widths
  //######################################################
  
  TLegend *legoHisto = new TLegend(0.15,0.75,0.39,0.89); 
  for(Int_t j=0; j < nOfFiles; j++) { 

    //###########################
    // Means
    //###########################

    //##################### phi ################## 

    c4->cd(1); 
    if(j==0){
      histoMeansdxy[j]->Draw("e1");
    }
    else  histoMeansdxy[j]->Draw("e1sames");
    
    legoHisto->AddEntry(histoMeansdxy[j],LegLabels[j]);
    legoHisto->SetFillColor(10);
    legoHisto->SetShadowColor(10);
    legoHisto->Draw();

    c5->cd(1);
    if(j==0){
      histoMeansdz[j]->Draw("e1");
    }
    else  histoMeansdz[j]->Draw("e1sames");
    legoHisto->Draw();

    //##################### Eta ################## 

    c4Eta->cd(1); 
    if(j==0){
      histoMeansdxyEta[j]->Draw("e1");
    }
    else  histoMeansdxyEta[j]->Draw("e1sames");
    
    legoHisto->Draw();

    c5Eta->cd(1);
    if(j==0){
      histoMeansdzEta[j]->Draw("e1");
    }
    else  histoMeansdzEta[j]->Draw("e1sames");
    legoHisto->Draw();


    //###########################
    // Sigmas
    //###########################
    
    //##################### Phi ################## 

    c4->cd(2);
    if(j==0){
      histoSigmasdxy[j]->Draw("e1");
    }
    else histoSigmasdxy[j]->Draw("e1sames");
    legoHisto->Draw();

    c5->cd(2);
    if(j==0){
      histoSigmasdz[j]->Draw("e1");
    }
    else histoSigmasdz[j]->Draw("e1sames");
    legoHisto->Draw();

    //##################### Eta ##################  

    c4Eta->cd(2);
    if(j==0){
      histoSigmasdxyEta[j]->Draw("e1");
    }
    else histoSigmasdxyEta[j]->Draw("e1sames");
    legoHisto->Draw();

    c5Eta->cd(2);
    if(j==0){
      histoSigmasdzEta[j]->Draw("e1");
    }
    else histoSigmasdzEta[j]->Draw("e1sames");
    legoHisto->Draw();
    
  }
  
  TString Canvas4Title ="HistoMeansSigmasDxy_"+label;
  TString Canvas4format=Canvas4Title+".png";  
  c4->SaveAs(Canvas4format);
  
  TString Canvas5Title ="HistoMeansSigmasDz_"+label;
  TString Canvas5format=Canvas5Title+".png"; 
  c5->SaveAs(Canvas5format);

  TString Canvas4TitleEta ="HistoMeansSigmasDxyEta_"+label;
  TString Canvas4formatEta=Canvas4TitleEta+".png";  
  c4Eta->SaveAs(Canvas4formatEta);
  
  TString Canvas5TitleEta ="HistoMeansSigmasDzEta_"+label;
  TString Canvas5formatEta=Canvas5TitleEta+".png"; 
  c5Eta->SaveAs(Canvas5formatEta);

}

//##########################################
// Fitting Function
//##########################################

void fitResiduals(TH1 *hist)
{
  //float fitResult(9999);
  if (!hist || hist->GetEntries() < 20) return;
  
  float mean  = hist->GetMean();
  float sigma = hist->GetRMS();
  
  TF1 func("tmp", "gaus", mean - 1.5*sigma, mean + 1.5*sigma); 
  if (0 == hist->Fit(&func,"QNR")) { // N: do not blow up file by storing fit!
    mean  = func.GetParameter(1);
    sigma = func.GetParameter(2);
    // second fit: three sigma of first fit around mean of first fit
    func.SetRange(mean - 2.*sigma, mean + 2.*sigma);
      // I: integral gives more correct results if binning is too wide
      // L: Likelihood can treat empty bins correctly (if hist not weighted...)
    if (0 == hist->Fit(&func, "Q0LR")) {
      if (hist->GetFunction(func.GetName())) { // Take care that it is later on drawn:
	hist->GetFunction(func.GetName())->ResetBit(TF1::kNotDraw);
      }
    }
  }
  return;
}
