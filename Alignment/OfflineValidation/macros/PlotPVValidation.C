/* 
 Macro to make PV Validation plots

 usage:
 $ root -l
 $ root [0] .L PlotPVValidation.C++
 $ root [1] PlotPVValidation("filename1.root=legendentry1,filename2.root=legendentry2",2,"<your TCut>",Bool_t useBS=false,Bool_t isNormal_=false,Bool_t verbose=false);

 - YourCut is a TCut (default ="")
 
 // M. Musich - INFN Torino

*/

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
#include <TPaveText.h>
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
#include <TDatime.h>
#include <TSpectrum.h>

// *------- constants ------* //
const Int_t nplots = 24; 
const Int_t nbins_ = 200;
Int_t _divs = nplots/2.;
Float_t _boundMin   = -0.5;
Float_t _boundSx    = (nplots/4.)-0.5;
Float_t _boundDx    = 3*(nplots/4.)-0.5;
Float_t _boundMax   = nplots-0.5;

ofstream outfile("FittedDeltaZ.txt");

TList *FileList;
TList *LabelList;
TFile *Target;

void MainLoop(TString namesandlabels,Int_t nOfFiles);

// *------- main function ------* //
void PlotPVValidation(TString namesandlabels,Int_t nOfFiles,TCut theCut="",Bool_t useBS=false,Bool_t isNormal_=false,Bool_t verbose=false);
void fitResiduals(TH1 *hist);
void doubleGausFitResiduals(TH1F *hist);

Double_t fULine(Double_t *x, Double_t *par);
Double_t fDLine(Double_t *x, Double_t *par);
void FitULine(TH1 *hist);
void FitDLine(TH1 *hist);

Double_t myCosine(Double_t *x, Double_t *par);
Double_t simplesine(Double_t *x, Double_t *par);
Double_t splitfunc(Double_t *x, Double_t *par);
Double_t splitfunconly(Double_t *x, Double_t *par);

void MakeNiceTF1Style(TF1 *f1,int color);
void MakeNiceHistoStyle(TH1 *hist);
void MakeNiceTrendPlotStyle(TH1 *hist);

Float_t calcFWHM(TH1F *hist);

/*--------------------------------------------------------------------*/
void MainLoop(TString namesandlabels,Int_t nOfFiles)
/*--------------------------------------------------------------------*/
{

  TObjArray *nameandlabelpairs = namesandlabels.Tokenize(",");
  TObjArray *aFileLegPair = TString(nameandlabelpairs->At(0)->GetName()).Tokenize("=");  
  
  TFile *f = TFile::Open(aFileLegPair->At(0)->GetName());
  TTree * evTree = (TTree *) f->Get("tree");
  unsigned int minRunNumber_=evTree->GetMinimum("RunNumber");
  unsigned int maxRunNumber_=evTree->GetMaximum("RunNumber");

  for(UInt_t i=minRunNumber_;i<maxRunNumber_;i++){
    char theRunCutstring[128];
    sprintf(theRunCutstring,"(RunNumber>=%.1i) && (RunNumber<%.1i)",i,i+1);
    std::cout<<theRunCutstring<<std::endl;
    TCut theRunCut=theRunCutstring;
    PlotPVValidation(namesandlabels,nOfFiles,theRunCut,false,false,false);
  }
}

/*--------------------------------------------------------------------*/
void PlotPVValidation(TString namesandlabels,Int_t nOfFiles, TCut theCut,Bool_t useBS,Bool_t isNormal_,Bool_t verbose)
/*--------------------------------------------------------------------*/
{

  TCut theDefaultCut= "hasRecVertex==1&&isGoodTrack==1";

  TH1::StatOverflows(kTRUE);
  gStyle->SetOptTitle(1);
  gStyle->SetOptStat("e");
  gStyle->SetPadTopMargin(0.12);
  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadLeftMargin(0.17);
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
  gStyle->SetNdivisions(510);

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
    cout<<"PlotPVValidation() label["<<j<<"]"<<LegLabels[j]<<endl;
    
  }

  TString label = LegLabels[nOfFiles-1];
  label.ReplaceAll(" ","_");
  
  
  /*----------------------------------*/
  TString _app;
  if(isNormal_) _app = "normal_";
  else _app="";
  /*----------------------------------*/

  TFile fileout("histos"+_app+label+".root", "new");
  
  //#################### Canvases #####################

  TCanvas *c1 = new TCanvas("c1","c1",800,550);
  c1->SetFillColor(10);  
  
  TCanvas *c2 = new TCanvas("c2","c2",1460,600);
  c2->SetFillColor(10);  
  c2->Divide(nplots/2,2);
  
  TCanvas *c3 = new TCanvas("c3","c3",1460,600);
  c3->SetFillColor(10);  
  c3->Divide(nplots/2,2);

  TCanvas *c2Eta = new TCanvas("c2Eta","c2Eta",1460,600);
  c2Eta->SetFillColor(10);  
  c2Eta->Divide(nplots/2,2);
  
  TCanvas *c3Eta = new TCanvas("c3Eta","c3Eta",1460,600);
  c3Eta->SetFillColor(10);  
  c3Eta->Divide(nplots/2,2);
   
  TCanvas *c4 = new TCanvas("c4","c4",1460,550);
  c4->SetFillColor(10);  
  c4->Divide(2,1);
  c4->cd(1)->SetBottomMargin(0.12);
  c4->cd(1)->SetLeftMargin(0.13);
  c4->cd(2)->SetBottomMargin(0.12);
  c4->cd(2)->SetLeftMargin(0.13);

  TCanvas *c4monly  = new TCanvas("c4monly","c4monly",730,550);
  c4monly->cd()->SetBottomMargin(0.12);
  c4monly->cd()->SetLeftMargin(0.13);
  c4monly->SetFillColor(10); 
  
  TCanvas *c5 = new TCanvas("c5","c5",1460,550);
  c5->SetFillColor(10);  
  c5->Divide(2,1);
  c5->cd(1)->SetBottomMargin(0.12);
  c5->cd(1)->SetLeftMargin(0.13);
  c5->cd(2)->SetBottomMargin(0.12);
  c5->cd(2)->SetLeftMargin(0.13);

  TCanvas *c5monly  = new TCanvas("c5monly","c5monly",730,550);
  c5monly->cd()->SetBottomMargin(0.12);
  c5monly->cd()->SetLeftMargin(0.13);
  c5monly->SetFillColor(10); 

  TCanvas *c4Eta = new TCanvas("c4Eta","c4Eta",1460,550);
  c4Eta->SetFillColor(10);  
  c4Eta->Divide(2,1);
  c4Eta->cd(1)->SetBottomMargin(0.12);
  c4Eta->cd(1)->SetLeftMargin(0.13);
  c4Eta->cd(2)->SetBottomMargin(0.12);
  c4Eta->cd(2)->SetLeftMargin(0.13);
  
  TCanvas *c5Eta = new TCanvas("c5Eta","c5Eta",1460,550);
  c5Eta->SetFillColor(10); 
  c5Eta->Divide(2,1);
  c5Eta->cd(1)->SetBottomMargin(0.12);
  c5Eta->cd(1)->SetLeftMargin(0.13);
  c5Eta->cd(2)->SetBottomMargin(0.12);
  c5Eta->cd(2)->SetLeftMargin(0.13);
               
  TLegend *lego = new TLegend(0.15,0.75,0.45,0.89);
  lego->SetFillColor(10);
  
  Int_t colors[8]={kBlack,kRed,kBlue,9,kGreen,5,8};
  //Int_t colors[8]={1,2,4,3,8,9,5};
  //Int_t colors[8]={6,2,4,1,3,8,9,5};
  //Int_t styles[8]={20,4,21,22,25,28,4,29}; 
  Int_t styles[8]={20,21,22,23,25,28,4,29}; 

  TH1F *histosdxy[nOfFiles][nplots];
  TH1F *histosdz[nOfFiles][nplots];

  TH1F *histosEtadxy[nOfFiles][nplots];
  TH1F *histosEtadz[nOfFiles][nplots];

  //############################
  // Canvas of bin residuals
  //############################
  
  float phipitch = (2*TMath::Pi())/nplots;
  float etapitch = 5./nplots;
  //cout<<"etapitch= "<<etapitch;
  
  for(Int_t i=0; i<nplots; i++){
    for(Int_t j=0; j < nOfFiles; j++){

      //for 12 bins
      //float phiF=(-TMath::Pi()+i*0.5235)*(180/TMath::Pi());
      //float phiL=(-TMath::Pi()+(i+1)*0.5235)*(180/TMath::Pi());
      //for 6 bins
      //float phiF=(-TMath::Pi()+i*1.047)*(180/TMath::Pi());
      //float phiL=(-TMath::Pi()+(i+1)*1.047)*(180/TMath::Pi());

      //for general number of bins
  
      float phiF=(-TMath::Pi()+i*phipitch)*(180/TMath::Pi());
      float phiL=(-TMath::Pi()+(i+1)*phipitch)*(180/TMath::Pi());

      float etaF=-2.5+i*etapitch;
      float etaL=-2.5+(i+1)*etapitch;
      
      float dxymax_phi,dzmax_phi, dxymax_eta, dzmax_eta;
      
      if(!isNormal_){
	if(useBS){
	  dxymax_phi=1500;
	  dzmax_phi=300000;
	  dxymax_eta=1500;
	  dzmax_eta=300000;
	} else {
	  dxymax_phi=3000;
	  dzmax_phi=3000;
	  dxymax_eta=3000;
	  dzmax_eta=3000;
	}
      } else {
	dxymax_phi=30;
	dzmax_phi=30;
	dxymax_eta=30;
	dzmax_eta=30;
      }

      //###################### vs phi #####################

      char histotitledxy[129];
      if(!isNormal_) sprintf(histotitledxy,"%.2f #circ<#varphi<%.2f #circ;d_{xy} [#mum];tracks",phiF,phiL);
      else sprintf(histotitledxy,"%.2f #circ<#varphi<%.2f #circ;d_{xy}/#sigma_{d_{xy}};tracks",phiF,phiL);
      char treeleg1[129];
      sprintf(treeleg1,"histodxyfile%i_plot%i",j,i);
      histosdxy[j][i] = new TH1F(treeleg1,histotitledxy,nbins_,-dxymax_phi,dxymax_phi);
      histosdxy[j][i]->SetLineColor(colors[j]);
      MakeNiceHistoStyle(histosdxy[j][i]);
    
      char histotitledz[129];
      if(!isNormal_) sprintf(histotitledz,"%.2f #circ<#varphi<%.2f #circ;d_{z} [#mum];tracks",phiF,phiL);
      else sprintf(histotitledz,"%.2f #circ<#varphi<%.2f #circ;d_{z}/#sigma_{d_{z}};tracks",phiF,phiL);
      char treeleg2[129];
      sprintf(treeleg2,"histodzfile%i_plot%i",j,i);
      histosdz[j][i] = new TH1F(treeleg2,histotitledz,nbins_,-dzmax_phi,dzmax_phi);
      histosdz[j][i]->SetLineColor(colors[j]);
      MakeNiceHistoStyle(histosdz[j][i]);
      
      //###################### vs eta ##################### 

      char histotitledxyeta[129];
      if(!isNormal_) sprintf(histotitledxyeta,"%.2f <#eta<%.2f ;d_{xy} [#mum];tracks",etaF,etaL);
      else sprintf(histotitledxyeta,"%.2f <#eta<%.2f;d_{xy}/#sigma_{d_{xy}};tracks",etaF,etaL);
      char treeleg1eta[129];
      sprintf(treeleg1eta,"histodxyEtafile%i_plot%i",j,i);
      histosEtadxy[j][i] = new TH1F(treeleg1eta,histotitledxyeta,nbins_,-dxymax_eta,dxymax_eta);
      histosEtadxy[j][i]->SetLineColor(colors[j]);
      MakeNiceHistoStyle(histosEtadxy[j][i]);
      
      char histotitledzeta[129];
      if(!isNormal_) sprintf(histotitledzeta,"%.2f <#eta<%.2f ;d_{z} [#mum];tracks",etaF,etaL);
      else sprintf(histotitledzeta,"%.2f <#eta<%.2f;d_{z}/#sigma_{d_{z}};tracks",etaF,etaL);
      char treeleg2eta[129];
      sprintf(treeleg2eta,"histodzEtafile%i_plot%i",j,i);
      histosEtadz[j][i] = new TH1F(treeleg2eta,histotitledzeta,nbins_,-dzmax_eta,dzmax_eta);
      histosEtadz[j][i]->SetLineColor(colors[j]);
      MakeNiceHistoStyle(histosEtadz[j][i]);
    }
  }
  
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
  
  TF1 *FitDzUp[nOfFiles];
  TF1 *FitDzDown[nOfFiles];
  
  TF1 *fleft[nOfFiles]; 
  TF1 *fright[nOfFiles];
  TF1 *fall[nOfFiles];  

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

  Double_t highedge=nplots-0.5;
  Double_t lowedge=-0.5;

  //#################### initialization limits ################################
  
  Double_t dxymean_phi, dxymean_eta, dxysigma_phi, dxysigma_eta, dzmean_phi, dzmean_eta, dzsigma_phi, dzsigma_eta ;

  if(!isNormal_){
    if(useBS){
      dxymean_phi=40; 
      dxymean_eta=40; 
      dxysigma_phi=310; 
      dxysigma_eta=510; 
      dzmean_phi=2000; 
      dzmean_eta=2000; 
      dzsigma_phi=100000; 
      dzsigma_eta=100000;
    } else {
      dxymean_phi=40; 
      dxymean_eta=40; 
      dxysigma_phi=260; 
      dxysigma_eta=260; 
      dzmean_phi=50; 
      dzmean_eta=160; 
      dzsigma_phi=410; 
      dzsigma_eta=1000;
    }
  } else {
    dxymean_phi=0.5; 
    dxymean_eta=0.5; 
    dxysigma_phi=2.; 
    dxysigma_eta=2.; 
    dzmean_phi=0.5; 
    dzmean_eta=0.5; 
    dzsigma_phi=2.; 
    dzsigma_eta=2.;
  }

  for(Int_t j=0; j < nOfFiles; j++) { 

    //################ phi #########################

    TString temptitle;
    if(!isNormal_) temptitle = "#LT d_{xy} #GT vs #varphi sector";
    else temptitle = "#LT d_{xy}/#sigma_{d_{xy}} #GT vs #varphi sector";

    char meanslegdxy[129];
    sprintf(meanslegdxy,"histomeansdxy_file%i",j);
    histoMeansdxy[j]  = new TH1F(meanslegdxy,temptitle,nplots,lowedge,highedge);
    histoMeansdxy[j]->SetLineColor(colors[j]);
    histoMeansdxy[j]->GetYaxis()->SetRangeUser(-dxymean_phi,dxymean_phi);
    histoMeansdxy[j]->SetMarkerStyle(styles[j]);
    histoMeansdxy[j]->SetMarkerColor(colors[j]);
    histoMeansdxy[j]->GetXaxis()->SetTitle("#varphi (sector) [degrees]");
    if(!isNormal_) histoMeansdxy[j]->GetYaxis()->SetTitle ("#LT d_{xy} #GT [#mum]");
    else histoMeansdxy[j]->GetYaxis()->SetTitle ("#LT d_{xy}/#sigma_{d_{xy}} #GT");
    MakeNiceTrendPlotStyle(histoMeansdxy[j]);
    
    temptitle.Clear();
    if(!isNormal_) temptitle = "#LT d_{z} #GT vs #varphi sector";
    else temptitle = "#LT d_{z}/#sigma_{d_{z}} #GT vs #varphi sector";

    char meanslegdz[129];
    sprintf(meanslegdz,"histomeansdz_file%i",j);
    histoMeansdz[j]   = new TH1F(meanslegdz,temptitle,nplots,lowedge,highedge); 
    histoMeansdz[j]->SetLineColor(colors[j]);
    histoMeansdz[j]->GetYaxis()->SetRangeUser(-dzmean_phi,dzmean_phi);
    histoMeansdz[j]->SetMarkerStyle(styles[j]);
    histoMeansdz[j]->SetMarkerColor(colors[j]);
    histoMeansdz[j]->GetXaxis()->SetTitle("#varphi (sector) [degrees]");
    if(!isNormal_) histoMeansdz[j]->GetYaxis()->SetTitle ("#LT d_{z} #GT [#mum]");
    else histoMeansdz[j]->GetYaxis()->SetTitle ("#LT d_{z}/#sigma_{d_{z}} #GT");
    MakeNiceTrendPlotStyle(histoMeansdz[j]);

    temptitle.Clear();
    if(!isNormal_) temptitle = "#sigma_{d_{xy}} vs #varphi sector";
    else temptitle = "width(d_{xy}/#sigma_{d_{xy}}) vs #varphi sector";

    char sigmaslegdxy[129];
    sprintf(sigmaslegdxy,"histosigmasdxy_file%i",j);
    histoSigmasdxy[j] = new TH1F(sigmaslegdxy,temptitle,nplots,lowedge,highedge);
    histoSigmasdxy[j]->SetLineColor(colors[j]);
    histoSigmasdxy[j]->GetYaxis()->SetRangeUser(0,dxysigma_phi);
    histoSigmasdxy[j]->SetMarkerStyle(styles[j]);
    histoSigmasdxy[j]->SetMarkerColor(colors[j]);
    histoSigmasdxy[j]->GetXaxis()->SetTitle("#varphi (sector) [degrees]");
    if(!isNormal_) histoSigmasdxy[j]->GetYaxis()->SetTitle ("#sigma_{d_{xy}} [#mum]");
    else histoSigmasdxy[j]->GetYaxis()->SetTitle ("width(d_{xy}/#sigma_{d_{xy}})");
    MakeNiceTrendPlotStyle(histoSigmasdxy[j]);

    temptitle.Clear();
    if(!isNormal_) temptitle = "#sigma_{d_{z}} vs #varphi sector";
    else temptitle = "width(d_{z}/#sigma_{d_{z}}) vs #varphi sector";

    char sigmaslegdz[129];
    sprintf(sigmaslegdz,"histosigmasdz_file%i",j);
    histoSigmasdz[j]  = new TH1F(sigmaslegdz,temptitle,nplots,lowedge,highedge);
    histoSigmasdz[j]->SetLineColor(colors[j]);
    histoSigmasdz[j]->GetYaxis()->SetRangeUser(0,dzsigma_phi);
    histoSigmasdz[j]->SetMarkerStyle(styles[j]);
    histoSigmasdz[j]->SetMarkerColor(colors[j]);
    histoSigmasdz[j]->GetXaxis()->SetTitle("#varphi (sector) [degrees]");
    if(!isNormal_) histoSigmasdz[j]->GetYaxis()->SetTitle ("#sigma_{d_{z}} [#mum]");
    else histoSigmasdz[j]->GetYaxis()->SetTitle ("width(d_{z}/#sigma_{d_{z}})");
    MakeNiceTrendPlotStyle(histoSigmasdz[j]);

    //################ eta #########################
  
    temptitle.Clear();
    if(!isNormal_) temptitle = "#LT d_{xy} #GT vs #eta sector"; 
    else temptitle = "#LT  d_{xy}/#sigma_{d_{xy}} #GT vs #eta sector";

    char meanslegdxyEta[129];
    sprintf(meanslegdxyEta,"histomeansEtadxy_file%i",j);
    histoMeansdxyEta[j]  = new TH1F(meanslegdxyEta,temptitle,nplots,lowedge,highedge);
    histoMeansdxyEta[j]->SetLineColor(colors[j]);
    histoMeansdxyEta[j]->GetYaxis()->SetRangeUser(-dxymean_eta,dxymean_eta);
    histoMeansdxyEta[j]->SetMarkerStyle(styles[j]);
    histoMeansdxyEta[j]->SetMarkerColor(colors[j]);
    histoMeansdxyEta[j]->GetXaxis()->SetTitle("#eta (sector)");
    if(!isNormal_) histoMeansdxyEta[j]->GetYaxis()->SetTitle ("#LT d_{xy} #GT [#mum]");
    else histoMeansdxyEta[j]->GetYaxis()->SetTitle ("#LT d_{xy}/#sigma_{d_{xy}} #GT");
    MakeNiceTrendPlotStyle(histoMeansdxyEta[j]);
   
    temptitle.Clear();
    if(!isNormal_) temptitle = "#LT d_{z} #GT vs #eta sector"; 
    else temptitle = "#LT d_{z}/#sigma_{d_{z}} #GT vs #eta sector";

    char meanslegdzEta[129];
    sprintf(meanslegdzEta,"histomeansEtadz_file%i",j);
    histoMeansdzEta[j]   = new TH1F(meanslegdzEta,temptitle,nplots,lowedge,highedge); 
    histoMeansdzEta[j]->SetLineColor(colors[j]);
    histoMeansdzEta[j]->GetYaxis()->SetRangeUser(-dzmean_eta,dzmean_eta);
    histoMeansdzEta[j]->SetMarkerStyle(styles[j]);
    histoMeansdzEta[j]->SetMarkerColor(colors[j]);
    histoMeansdzEta[j]->GetXaxis()->SetTitle("#eta (sector)");
    if(!isNormal_) histoMeansdzEta[j]->GetYaxis()->SetTitle ("#LT d_{z} #GT [#mum]");
    else histoMeansdzEta[j]->GetYaxis()->SetTitle ("#LT d_{z}/#sigma_{d_{z}} #GT");
    MakeNiceTrendPlotStyle(histoMeansdzEta[j]);

    temptitle.Clear();
    if(!isNormal_) temptitle = "#sigma_{d_{xy}} vs #eta sector"; 
    else temptitle = "width(d_{xy}/#sigma_{d_{xy}}) vs #eta sector";

    char sigmaslegdxyEta[129];
    sprintf(sigmaslegdxyEta,"histosigmasEtadxy_file%i",j);
    histoSigmasdxyEta[j] = new TH1F(sigmaslegdxyEta,temptitle,nplots,lowedge,highedge);
    histoSigmasdxyEta[j]->SetLineColor(colors[j]);
    histoSigmasdxyEta[j]->GetYaxis()->SetRangeUser(0,dxysigma_eta);
    histoSigmasdxyEta[j]->SetMarkerStyle(styles[j]);
    histoSigmasdxyEta[j]->SetMarkerColor(colors[j]);
    histoSigmasdxyEta[j]->GetXaxis()->SetTitle("#eta (sector)");
    if(!isNormal_) histoSigmasdxyEta[j]->GetYaxis()->SetTitle ("#sigma_{d_{xy}} [#mum]");
    else histoSigmasdxyEta[j]->GetYaxis()->SetTitle ("width(d_{xy}/#sigma_{d_{xy}})");
    MakeNiceTrendPlotStyle(histoSigmasdxyEta[j]);

    temptitle.Clear();
    if(!isNormal_) temptitle = "#sigma_{d_{z}} vs #eta sector"; 
    else temptitle = "width(d_{z}/#sigma_{d_{z}}) vs #eta sector";

    char sigmaslegdzEta[129];
    sprintf(sigmaslegdzEta,"histosigmasEtadz_file%i",j);
    histoSigmasdzEta[j]  = new TH1F(sigmaslegdzEta,temptitle,nplots,lowedge,highedge);
    histoSigmasdzEta[j]->SetLineColor(colors[j]);
    histoSigmasdzEta[j]->GetYaxis()->SetRangeUser(0,dzsigma_eta);
    histoSigmasdzEta[j]->SetMarkerStyle(styles[j]);
    histoSigmasdzEta[j]->SetMarkerColor(colors[j]);
    histoSigmasdzEta[j]->GetXaxis()->SetTitle("#eta (sector)");
    if(!isNormal_) histoSigmasdzEta[j]->GetYaxis()->SetTitle ("#sigma_{d_{z}} [#mum]");
    else histoSigmasdzEta[j]->GetYaxis()->SetTitle ("width(d_{z}/#sigma_{d_{z}})");
    MakeNiceTrendPlotStyle(histoSigmasdzEta[j]);
  }

  
   //Histos maxima
   
   double hmaxDxyphi=0.;
   double hmaxDxyeta=0.;
   double hmaxDzphi=0.;
   double hmaxDzeta=0.;

   //######################## dxy Residuals #################################
   
   for(Int_t j=0; j < nOfFiles; j++) { 
     std::cout<<"PlotPVValidation() Running on file: "<<LegLabels[j]<<std::endl;
     for(Int_t i=0; i<nplots; i++){
       if(verbose){ 
	 std::cout<<"PlotPVValidation() Fitting the : "<<i<<"/"<<nplots<<" bin ";
	 TDatime *time = new TDatime();
	 time->Print();
	 //std::cout<<std::endl;
       }

       //################### phi ########################
       
       char thePhiCutstring[128];
       sprintf(thePhiCutstring,"(phi>-TMath::Pi()+%.1i*(%.3f))&&(phi<-TMath::Pi()+%.1i*(%.3f))",i,phipitch,i+1,phipitch);
       TCut thePhiCut = thePhiCutstring;
       //cout<<thePhiCutstring<<endl;
       
       char treeleg3[129];

       if(!isNormal_){
	 if(useBS){
	   sprintf(treeleg3,"dxyBs*10000>>histodxyfile%i_plot%i",j,i);
	 } else {
	   sprintf(treeleg3,"dxyFromMyVertex*10000>>histodxyfile%i_plot%i",j,i);
	 } 
       } else {
	 sprintf(treeleg3,"IPTsigFromMyVertex>>histodxyfile%i_plot%i",j,i);
       }

       char phipositionString[129];
       float_t phiInterval = (360.)/nplots;
       float phiposition = (-180+i*phiInterval)+(phiInterval/2);
       sprintf(phipositionString,"%.f",phiposition);
       //cout<<"phiposition: "<<phiposition<<" phipositionString: "<<phipositionString<<endl;

       c2->cd(i+1);
       if(j==0){
	 trees[j]->Draw(treeleg3,theCut&&theDefaultCut&&thePhiCut);  
	 hmaxDxyphi=histosdxy[0][i]->GetMaximum();
       }
       else{ 
	 trees[j]->Draw(treeleg3,theCut&&theDefaultCut&&thePhiCut,"sames"); 
	 
	 if(histosdxy[j][i]->GetMaximum() >  hmaxDxyphi){
	   hmaxDxyphi = histosdxy[j][i]->GetMaximum();
	 }
       }
       
       histosdxy[0][i]->GetYaxis()->SetRangeUser(0.01,hmaxDxyphi*1.10);
       histosdxy[0][i]->Draw("sames");
       
       if(j!=0){
	 histosdxy[j][i]->Draw("sames");
       }

       histosdxy[j][i]->Write();

       fitResiduals(histosdxy[j][i]);   
       tmpsdxy[j][i] = (TF1*)histosdxy[j][i]->GetListOfFunctions()->FindObject("tmp");   
       if(tmpsdxy[j][i])  {
	 tmpsdxy[j][i]->SetLineColor(colors[j]);
	 tmpsdxy[j][i]->SetLineWidth(2); 
	 tmpsdxy[j][i]->Draw("same"); 
       
	 meansdxy[j][i]=(tmpsdxy[j][i])->GetParameter(1);
	 sigmasdxy[j][i]=(tmpsdxy[j][i])->GetParameter(2);
	 
	 meansdxyError[j][i]=(tmpsdxy[j][i])->GetParError(1);
	 sigmasdxyError[j][i]=(tmpsdxy[j][i])->GetParError(2); 
	 
	 histoMeansdxy[j]->SetBinContent(i+1,meansdxy[j][i]); 
	 histoMeansdxy[j]->SetBinError(i+1,meansdxyError[j][i]); 
	 histoMeansdxy[j]->GetXaxis()->SetBinLabel(i+1,phipositionString);
	 histoSigmasdxy[j]->SetBinContent(i+1,sigmasdxy[j][i]);
	 histoSigmasdxy[j]->SetBinError(i+1,sigmasdxyError[j][i]); 
	 histoSigmasdxy[j]->GetXaxis()->SetBinLabel(i+1,phipositionString);
       }

       else{
	 histoMeansdxy[j]->SetBinContent(i+1,0.); 
	 histoMeansdxy[j]->SetBinError(i+1,0.);
	 histoMeansdxy[j]->GetXaxis()->SetBinLabel(i+1,phipositionString);
	 histoSigmasdxy[j]->SetBinContent(i+1,0.);
	 histoSigmasdxy[j]->SetBinError(i+1,0.); 
	 histoSigmasdxy[j]->GetXaxis()->SetBinLabel(i+1,phipositionString); 
       }
       
       c2->Draw("sames");   
       c2->cd(i+1);
       
       statObjsdxy[j][i] = histosdxy[j][i]->GetListOfFunctions()->FindObject("stats");
       statsdxy[j][i] = static_cast<TPaveStats*>(statObjsdxy[j][i]);
       statsdxy[j][i]->SetLineColor(colors[j]);
       statsdxy[j][i]->SetTextColor(colors[j]);
       statsdxy[j][i]->SetShadowColor(10);
       statsdxy[j][i]->SetFillColor(10);
       statsdxy[j][i]->SetX1NDC(0.66);
       statsdxy[j][i]->SetY1NDC(0.80-j*0.19);
       statsdxy[j][i]->SetX2NDC(0.99);
       statsdxy[j][i]->SetY2NDC(0.99-j*0.19);
       statsdxy[j][i]->Draw("same");   
  
       //################### eta ########################
       
       char theEtaCutstring[128];
       sprintf(theEtaCutstring,"(eta>-2.5+%.1i*(%.3f))&&(eta<-2.5+%.1i*(%.3f))",i,etapitch,i+1,etapitch);
       TCut theEtaCut = theEtaCutstring;
       //cout<<theEtaCutstring<<endl;
       
       char treeleg3Eta[129];

       if(!isNormal_){
	 if(useBS){
	   sprintf(treeleg3Eta,"dxyBs*10000>>histodxyEtafile%i_plot%i",j,i);
	 } else {
	   sprintf(treeleg3Eta,"dxyFromMyVertex*10000>>histodxyEtafile%i_plot%i",j,i);
	 }
       } else {
	 sprintf(treeleg3Eta,"IPTsigFromMyVertex>>histodxyEtafile%i_plot%i",j,i);
       }      

       char etapositionString[129];
       float etaposition = (-2.5+i*(5./nplots))+((5./nplots)/2);
       sprintf(etapositionString,"%.1f",etaposition);
       //cout<<"etaposition: "<<etaposition<<" etapositionString: "<<etapositionString<<endl;
       
       c2Eta->cd(i+1);
       if(j==0){
	 trees[j]->Draw(treeleg3Eta,theCut&&theDefaultCut&&theEtaCut);
	 hmaxDxyeta=histosEtadxy[0][i]->GetMaximum();  
       }
       else{ 
	 trees[j]->Draw(treeleg3Eta,theCut&&theDefaultCut&&theEtaCut,"sames"); 
	 if(histosEtadxy[j][i]->GetMaximum() >  hmaxDxyeta){
	   hmaxDxyeta = histosEtadxy[j][i]->GetMaximum();
	 }
       }

       histosEtadxy[0][i]->GetYaxis()->SetRangeUser(0.01,hmaxDxyeta*1.10);
       histosEtadxy[0][i]->Draw("sames");
       
       if(j!=0){
	 histosEtadxy[j][i]->Draw("sames");
       }

       histosEtadxy[j][i]->Write();

       fitResiduals(histosEtadxy[j][i]);   
       tmpsdxyEta[j][i] = (TF1*)histosEtadxy[j][i]->GetListOfFunctions()->FindObject("tmp");   
       if(tmpsdxyEta[j][i])  {
	 tmpsdxyEta[j][i]->SetLineColor(colors[j]);
	 tmpsdxyEta[j][i]->SetLineWidth(2); 
	 
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
       } else{
	 histoMeansdxyEta[j]->SetBinContent(i+1,0.); 
	 histoMeansdxyEta[j]->SetBinError(i+1,0.); 	 
	 histoMeansdxyEta[j]->GetXaxis()->SetBinLabel(i+1,etapositionString);
	 histoSigmasdxyEta[j]->SetBinContent(i+1,0.);
	 histoSigmasdxyEta[j]->SetBinError(i+1,0.);
	 histoSigmasdxyEta[j]->GetXaxis()->SetBinLabel(i+1,etapositionString);
       }

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
       
       //######################## dz Residuals #################################
       
       char treeleg4[129];
       
       if(!isNormal_){
	 if(useBS){
	   sprintf(treeleg4,"dzBs*10000>>histodzfile%i_plot%i",j,i);
	 } else {
	   sprintf(treeleg4,"dzFromMyVertex*10000>>histodzfile%i_plot%i",j,i);
	 }
       } else {
	 sprintf(treeleg4,"IPLsigFromMyVertex>>histodzfile%i_plot%i",j,i);
       }        

       c3->cd(i+1);
       if(j==0){
	 trees[j]->Draw(treeleg4,theCut&&theDefaultCut&&thePhiCut);
	 hmaxDzphi=histosdz[0][i]->GetMaximum();  
       }
       
       else{ 
	 trees[j]->Draw(treeleg4,theCut&&theDefaultCut&&thePhiCut,"sames"); 
	 if(histosdz[j][i]->GetMaximum() >  hmaxDzphi){
	   hmaxDzphi = histosdz[j][i]->GetMaximum();
	 }
       }
       
       histosdz[0][i]->GetYaxis()->SetRangeUser(0.01,hmaxDzphi*1.10);
       histosdz[0][i]->Draw("sames");
       
       if(j!=0){
	 histosdz[j][i]->Draw("sames");
       }
       
       histosdz[j][i]->Write();

       fitResiduals(histosdz[j][i]);   
       tmpsdz[j][i] = (TF1*)histosdz[j][i]->GetListOfFunctions()->FindObject("tmp");   
       if(tmpsdz[j][i])  {
	 tmpsdz[j][i]->SetLineColor(colors[j]);
	 tmpsdz[j][i]->SetLineWidth(2);
	 tmpsdz[j][i]->Draw("same");  
	 	 
	 meansdz[j][i]=(tmpsdz[j][i])->GetParameter(1);
	 sigmasdz[j][i]=(tmpsdz[j][i])->GetParameter(2);
	 
	 meansdzError[j][i]=(tmpsdz[j][i])->GetParError(1);
	 sigmasdzError[j][i]=(tmpsdz[j][i])->GetParError(2);
	 
	 histoMeansdz[j]->SetBinContent(i+1,meansdz[j][i]); 
	 histoMeansdz[j]->SetBinError(i+1,meansdzError[j][i]); 
	 histoMeansdz[j]->GetXaxis()->SetBinLabel(i+1,phipositionString);
	 histoSigmasdz[j]->SetBinContent(i+1,sigmasdz[j][i]);
	 histoSigmasdz[j]->SetBinError(i+1,sigmasdzError[j][i]);
	 histoSigmasdz[j]->GetXaxis()->SetBinLabel(i+1,phipositionString);
       }
       
       else{
	 histoMeansdz[j]->SetBinContent(i+1,0.); 
	 histoMeansdz[j]->SetBinError(i+1,0.); 
	 histoMeansdz[j]->GetXaxis()->SetBinLabel(i+1,phipositionString);
	 histoSigmasdz[j]->SetBinContent(i+1,0.);
	 histoSigmasdz[j]->SetBinError(i+1,0.);
	 histoSigmasdz[j]->GetXaxis()->SetBinLabel(i+1,phipositionString);
       }
       
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
              
       char treeleg4Eta[129];

       if(!isNormal_){
	 if(useBS){
	   sprintf(treeleg4Eta,"dzBs*10000>>histodzEtafile%i_plot%i",j,i);	
	 } else {
	   sprintf(treeleg4Eta,"dzFromMyVertex*10000>>histodzEtafile%i_plot%i",j,i);
	 } 
       } else {
	 sprintf(treeleg4Eta,"IPLsigFromMyVertex>>histodzEtafile%i_plot%i",j,i);
       }      
       
       c3Eta->cd(i+1);
       if(j==0){
	 trees[j]->Draw(treeleg4Eta,theCut&&theDefaultCut&&theEtaCut);
	 hmaxDzeta=histosEtadz[0][i]->GetMaximum();    
       }
       else{ trees[j]->Draw(treeleg4Eta,theCut&&theDefaultCut&&theEtaCut,"sames");
	 if(histosEtadz[j][i]->GetMaximum() >  hmaxDzeta){
	   hmaxDzeta = histosdz[j][i]->GetMaximum();
	 }
       }
       
       histosEtadz[0][i]->GetYaxis()->SetRangeUser(0.01,hmaxDzeta*1.10);
       histosEtadz[0][i]->Draw("sames");
       
       if(j!=0){
	 histosEtadz[j][i]->Draw("sames");
       }
       
       histosEtadz[j][i]->Write();

       fitResiduals(histosEtadz[j][i]);   
       tmpsdzEta[j][i] = (TF1*)histosEtadz[j][i]->GetListOfFunctions()->FindObject("tmp");   
       if(tmpsdzEta[j][i])  {
	 tmpsdzEta[j][i]->SetLineColor(colors[j]);
	 tmpsdzEta[j][i]->SetLineWidth(2); 
	 
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
	 
       }  else {
	 histoMeansdzEta[j]->SetBinContent(i+1,0.); 
	 histoMeansdzEta[j]->SetBinError(i+1,0.); 
	 histoMeansdzEta[j]->GetXaxis()->SetBinLabel(i+1,etapositionString);
	 histoSigmasdzEta[j]->SetBinContent(i+1,0.);
	 histoSigmasdzEta[j]->SetBinError(i+1,0.);
	 histoSigmasdzEta[j]->GetXaxis()->SetBinLabel(i+1,etapositionString);
       }
       
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
   
   TString Canvas2Title =_app+"VertexDxyResidualsPhiBin_"+label;
   TString Canvas2formatpng=Canvas2Title+".png"; 
   TString Canvas2formateps=Canvas2Title+".pdf"; 
   c2->SaveAs(Canvas2formatpng);
   c2->SaveAs(Canvas2formateps);
   c2->Write();
  
   TString Canvas2TitleEta =_app+"VertexDxyResidualsEtaBin_"+label;
   TString Canvas2formatEtapng=Canvas2TitleEta+".png"; 
   TString Canvas2formatEtaeps=Canvas2TitleEta+".pdf"; 
   c2Eta->SaveAs(Canvas2formatEtapng);
   c2Eta->SaveAs(Canvas2formatEtaeps);
   c2Eta->Write();
  
   TString Canvas3Title =_app+"VertexDzResidualsPhiBin_"+label;
   TString Canvas3formatpng=Canvas3Title+".png";
   TString Canvas3formateps=Canvas3Title+".pdf";  
   c3->SaveAs(Canvas3formatpng);
   c3->SaveAs(Canvas3formateps);
   c3->Write();
 
   TString Canvas3TitleEta =_app+"VertexDzResidualsEtaBin_"+label;
   TString Canvas3formatEtapng=Canvas3TitleEta+".png";
   TString Canvas3formatEtaeps=Canvas3TitleEta+".pdf";
   c3Eta->SaveAs(Canvas3formatEtapng);
   c3Eta->SaveAs(Canvas3formatEtaeps);
   c3Eta->Write();
   
   TPaveText *pt = new TPaveText(0.63,0.73,0.8,0.83,"NDC");
   pt->SetFillColor(10);
   pt->SetTextColor(1);
   //pt->SetTextSize(0.05);
   pt->SetTextFont(42);
   pt->SetTextAlign(11);
   TText *text1 = pt->AddText("Tk Alignment 2012");
   text1->SetTextSize(0.05);
   TText *text2 = pt->AddText("#sqrt{s}=8 TeV");
   text2->SetTextSize(0.05); 
   
   //######################################################
   //  Histograms Means and Widths
   //######################################################
   
   TLegend *legoHisto = new TLegend(0.20,0.65,0.40,0.85); 
   legoHisto->SetLineColor(10);
   legoHisto->SetTextSize(0.035);
   legoHisto->SetTextFont(42);
   
   TLegend *legoHistoSeparation = new TLegend(0.14,0.67,0.58,0.87); 
   legoHistoSeparation-> SetNColumns(2);
   legoHistoSeparation->SetLineColor(10);
   legoHistoSeparation->SetTextFont(42);
   legoHistoSeparation->SetFillColor(10);
   legoHistoSeparation->SetTextSize(0.04);
   legoHistoSeparation->SetShadowColor(10);
   
   TLegend *legoHistoSine = new TLegend(0.14,0.16,0.58,0.34); 
   legoHistoSine-> SetNColumns(2);
   legoHistoSine->SetLineColor(10);
   legoHistoSine->SetTextFont(42);
   legoHistoSine->SetFillColor(10);
   legoHistoSine->SetTextSize(0.03);
   legoHistoSine->SetShadowColor(10);

   TF1 *func[nOfFiles];
   TF1 *func2[nOfFiles];

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
     pt->Draw("sames");

    c5->cd(1);
    if(j==0){
      histoMeansdz[j]->Draw("e1");
    }
    else  histoMeansdz[j]->Draw("e1sames");
    legoHisto->Draw();
    pt->Draw("sames");

    //#########################################
    // 
    // Plots with means only
    //
    //#########################################

    c4monly->cd();

    if(j==0){
      histoMeansdxy[j]->SetMarkerSize(1.1);
      histoMeansdxy[j]->Draw("e1");
    }
    else  histoMeansdxy[j]->Draw("e1sames");

    TCanvas *theNewCanvas3 = new TCanvas("NewCanvas3","Fitting Canvas 3",800,600);
    TH1F *hnewDxy = (TH1F*)histoMeansdxy[j]->Clone("hnewDxy");
    
    func[j] = new TF1(Form("myfunc%i",j),splitfunc,_boundMin,_boundMax,4);
    func[j]->SetParameters(5,5,5,5);
    func[j]->SetParLimits(0,-80.,80.);
    func[j]->SetParLimits(1,-80.,80.);
    func[j]->SetParLimits(2,-80.,80.);
    func[j]->SetParLimits(3,-80.,80.);
    func[j]->SetParNames("A","B","k_{+}","k_{-}");
    func[j]->SetLineColor(colors[j]);
    func[j]->SetNpx(1000);
    
    func2[j] = new TF1(Form("myfunc2%i",j),simplesine,_boundMin,_boundMax,2);
    //func2[j] = new TF1(Form("myfunc2%i",j),myCosine,_boundMin,_boundMax,2);
    func2[j]->SetParameters(5,5);
    func2[j]->SetParLimits(0,-80.,80.);
    func2[j]->SetParLimits(1,-80.,80.);
    func2[j]->SetParNames("A","B");
    func2[j]->SetLineColor(colors[j]);
    func2[j]->SetNpx(1000);
     
    theNewCanvas3->cd();
    hnewDxy->Draw();
    hnewDxy->Fit(func2[j]);
    
    c4monly->cd();
    func2[j]->Draw("same");
    histoMeansdxy[j]->Draw("e1sames");
    
    TString COUT1 = Form("#splitline{"+LegLabels[j]+"}{#splitline{d_{x}=%.1f #pm %.1f #mum}{d_{y}=%.1f #pm %.1f #mum}}",func2[j]->GetParameter(0),func2[j]->GetParError(0),func2[j]->GetParameter(1),func2[j]->GetParError(0));
    
    legoHistoSine->AddEntry(histoMeansdxy[j],COUT1);
    legoHistoSine->Draw();

    TPaveText *pi = new TPaveText(0.14,0.71,0.54,0.84,"NDC");
    pi->SetFillColor(10);
    pi->SetTextColor(1);
    pi->SetTextSize(0.04);
    pi->SetTextFont(42);
    pi->SetTextAlign(11);
    TText *textPi1 = pi->AddText("Fitting function:");
    textPi1->SetTextSize(0.05);
    // TText *textPi2 = pi->AddText("d_{xy} (#varphi) = #ltbar #splitline{|#varphi|<#pi/2     A cos #varphi + B sin #varphi + k_{+} (1- sin #varphi) }{|#varphi|>#pi/2       - A cos #varphi  - B sin #varphi - k_{-} (1- sin#varphi) } ");
    //   TText *textPi2 = pi->AddText(Form(" #splitline{d_{xy} (#varphi) = A sin #varphi + B cos #varphi }{A=%.1f  B=%.1f}",func2[j]->GetParameter(0),func2[j]->GetParameter(1)));
    TText *textPi2 = pi->AddText("d_{xy} (#varphi) = d_{x} sin #varphi + d_{y} cos #varphi");
    textPi2->SetTextSize(0.04); 
    textPi2->SetTextSize(0.04); 

    pi->Draw("sames");

    //legoHisto->Draw();
    pt->Draw("sames");

    c5monly->cd();
    
    Double_t deltaZ(0);
    Double_t sigmadeltaZ(-1);
    
    if(j==0){
      histoMeansdz[j]->SetMarkerSize(1.1);
      histoMeansdz[j]->Draw("e1");
    }
    else  histoMeansdz[j]->Draw("e1sames");
   
    TCanvas *theNewCanvas2 = new TCanvas("NewCanvas2","Fitting Canvas 2",800,600);
    theNewCanvas2->Divide(2,1);

    TH1F *hnewUp = (TH1F*)histoMeansdz[j]->Clone("hnewUp");
    TH1F *hnewDown = (TH1F*)histoMeansdz[j]->Clone("hnewDown");
    
    fleft[j] = new TF1(Form("fleft_%i",j),fULine,_boundMin,_boundSx,1);
    fright[j] = new TF1(Form("fright_%i",j),fULine,_boundDx,_boundMax,1);
    fall[j] = new TF1(Form("fall_%i",j),fDLine,_boundSx,_boundDx,1);
    
    FitULine(hnewUp);  
    FitDzUp[j]   = (TF1*)hnewUp->GetListOfFunctions()->FindObject("lineUp"); 
    if(FitDzUp[j]){
      fleft[j]->SetParameters(FitDzUp[j]->GetParameters());
      fleft[j]->SetParErrors(FitDzUp[j]->GetParErrors());
      hnewUp->GetListOfFunctions()->Add(fleft[j]);
      fright[j]->SetParameters(FitDzUp[j]->GetParameters());
      fright[j]->SetParErrors(FitDzUp[j]->GetParErrors());
      hnewUp->GetListOfFunctions()->Add(fright[j]);
      FitDzUp[j]->Delete();

      theNewCanvas2->cd(1);
      MakeNiceTF1Style(fright[j],colors[j]);
      MakeNiceTF1Style(fleft[j],colors[j]);
      fright[j]->Draw("same");
      fleft[j]->Draw("same");
    }
    
    FitDLine(hnewDown);  
    FitDzDown[j] = (TF1*)hnewDown->GetListOfFunctions()->FindObject("lineDown");    
    
    if(FitDzDown[j]){
      fall[j]->SetParameters(FitDzDown[j]->GetParameters());
      fall[j]->SetParErrors(FitDzDown[j]->GetParErrors());
      hnewDown->GetListOfFunctions()->Add(fall[j]);
      FitDzDown[j]->Delete();
      
      theNewCanvas2->cd(2);
      MakeNiceTF1Style(fall[j],colors[j]);
      fall[j]->Draw("same");
      c5monly->cd();
      fright[j]->Draw("sames");
      fleft[j]->Draw("same");
      fall[j]->Draw("same");
    }
    
    if(j==nOfFiles-1){
      theNewCanvas2->Close();
    }

    deltaZ=(fright[j]->GetParameter(0) - fall[j]->GetParameter(0));
    sigmadeltaZ=TMath::Sqrt(fright[j]->GetParError(0)*fright[j]->GetParError(0) + fall[j]->GetParError(0)*fall[j]->GetParError(0));
    TString COUT2 = Form("#splitline{"+LegLabels[j]+"}{#Delta z = %.f #pm %.f #mum}",deltaZ,sigmadeltaZ);
    
    if(j==nOfFiles-1){ 
      outfile <<deltaZ<<"|"<<sigmadeltaZ<<endl;
    }

    legoHistoSeparation->AddEntry(histoMeansdxy[j],COUT2);
    legoHistoSeparation->Draw();

    pt->Draw("sames");

    //##################### Eta ################## 

    c4Eta->cd(1); 
    if(j==0){
      histoMeansdxyEta[j]->Draw("e1");
    }
    else  histoMeansdxyEta[j]->Draw("e1sames");
    
    legoHisto->Draw();
    pt->Draw("sames");

    c5Eta->cd(1);
    if(j==0){
      histoMeansdzEta[j]->Draw("e1");
    }
    else  histoMeansdzEta[j]->Draw("e1sames");
    legoHisto->Draw();
    pt->Draw("sames");

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
    pt->Draw("sames");

    c5->cd(2);
    if(j==0){
      histoSigmasdz[j]->Draw("e1");
    }
    else histoSigmasdz[j]->Draw("e1sames");
    legoHisto->Draw();
    pt->Draw("sames");

    //##################### Eta ##################  

    c4Eta->cd(2);
    if(j==0){
      histoSigmasdxyEta[j]->Draw("e1");
    }
    else histoSigmasdxyEta[j]->Draw("e1sames");
    legoHisto->Draw();
    pt->Draw("sames");

    c5Eta->cd(2);
    if(j==0){
      histoSigmasdzEta[j]->Draw("e1");
    }
    else histoSigmasdzEta[j]->Draw("e1sames");
    legoHisto->Draw();
    pt->Draw("sames");
      
  }
  
  TString Canvas4Title =_app+"HistoMeansSigmasDxy_"+label;
  TString Canvas4formatpng=Canvas4Title+".png"; 
  TString Canvas4formatpdf=Canvas4Title+".pdf"; 
  c4->SaveAs(Canvas4formatpng);
  c4->SaveAs(Canvas4formatpdf);
  c4->Write();
  
  TString Canvas4Titlemonly =_app+"HistoMeansDxy_"+label;
  TString Canvas4formatpngmonly=Canvas4Titlemonly+".png"; 
  TString Canvas4formatpdfmonly=Canvas4Titlemonly+".pdf"; 
  TString Canvas4formatrootmonly=Canvas4Titlemonly+".root"; 
  c4monly->SaveAs(Canvas4formatpngmonly);
  c4monly->SaveAs(Canvas4formatpdfmonly);
  c4monly->Write();
 
  TString Canvas5Title =_app+"HistoMeansSigmasDz_"+label;
  TString Canvas5formatpng=Canvas5Title+".png"; 
  TString Canvas5formatpdf=Canvas5Title+".pdf";
  c5->SaveAs(Canvas5formatpng);
  c5->Write();
  c5->SaveAs(Canvas5formatpdf);

  TString Canvas5Titlemonly =_app+"HistoMeansDzFit_"+label;
  TString Canvas5formatpngmonly=Canvas5Titlemonly+".png"; 
  TString Canvas5formatpdfmonly=Canvas5Titlemonly+".pdf"; 
  TString Canvas5formatrootmonly=Canvas5Titlemonly+".root"; 
  c5monly->SaveAs(Canvas5formatpngmonly);
  c5monly->SaveAs(Canvas5formatpdfmonly);
  c5monly->Write();
 
  TString Canvas4TitleEta =_app+"HistoMeansSigmasDxyEta_"+label;
  TString Canvas4formatEtapng=Canvas4TitleEta+".png"; 
  TString Canvas4formatEtapdf=Canvas4TitleEta+".pdf"; 
  c4Eta->SaveAs(Canvas4formatEtapng);
  c4Eta->SaveAs(Canvas4formatEtapdf);
  c4Eta->Write();
  
  TString Canvas5TitleEta =_app+"HistoMeansSigmasDzEta_"+label;
  TString Canvas5formatEtapng=Canvas5TitleEta+".png";
  TString Canvas5formatEtapdf=Canvas5TitleEta+".pdf";
  c5Eta->SaveAs(Canvas5formatEtapng);
  c5Eta->SaveAs(Canvas5formatEtapdf);
  c5Eta->Write();
  
  outfile.close();

}

//##########################################
// Fitting Function
//##########################################

/*--------------------------------------------------------------------*/
void fitResiduals(TH1 *hist)
/*--------------------------------------------------------------------*/
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
    func.SetRange(mean - 2*sigma, mean + 2*sigma);
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

/*--------------------------------------------------------------------*/
void olddoubleGausFitResiduals(TH1 *hist)
/*--------------------------------------------------------------------*/
{
  //float fitResult(9999);
  // if (!hist || hist->GetEntries() < 20) return;
  
  float mean  = hist->GetMean();
  float sigma = hist->GetRMS();
  float interval = (hist->GetXaxis()->GetBinLowEdge(hist->GetNbinsX()+1) - hist->GetXaxis()->GetBinLowEdge(0))/2.;

  Double_t par[6];
  TF1 func1("gaus1", "gaus", mean - 1.5*sigma, mean + 1.5*sigma); 
  // TF1 func2("gaus2", "gaus", mean - 2.5*sigma, mean + 2.5*sigma); 
  TF1 func2("gaus2", "gaus", mean - interval, mean + interval); 
  TF1 func("tmp", "gaus(0)+gaus(3)", mean - 3*sigma, mean + 3*sigma); 
  
  hist->Fit(&func1,"QNR");
  hist->Fit(&func2,"QNR+");
  func1.GetParameters(&par[0]);
  func2.GetParameters(&par[3]);
  // cout<<"partials fit done!"<<endl;
  
  if(hist->GetEntries()>20) {
    
    // cout<<"histo entries: "<<hist->GetEntries()<<endl; 

    func.SetParameters(par);
    func.SetParLimits(1,par[1] + 1*par[2],par[1] + 1*par[2]);
    func.SetParLimits(3,par[1] + 1*par[2],par[1] + 1*par[2]);
    
    if (0 == hist->Fit(&func,"QNR")) { // N: do not blow up file by storing fit!
      
      // cout<<"before extracting parameters"<<endl;
      
      mean  = func.GetParameter(1);
      sigma = func.GetParameter(5);
      
      // cout<<"first total fit done!"<<endl;
      
      // second fit: three sigma of first fit around mean of first fit
      
      func.SetRange(mean - 3*sigma, mean + 3*sigma);
      // I: integral gives more correct results if binning is too wide
      // L: Likelihood can treat empty bins correctly (if hist not weighted...)
      if (0 == hist->Fit(&func, "Q0LR")) {
	if (hist->GetFunction(func.GetName())) { // Take care that it is later on drawn:
	  hist->GetFunction(func.GetName())->ResetBit(TF1::kNotDraw);
	}
 
	//cout<<"fit done!"<<endl;
	
      }
    }
    
  } else {
    cout<<"Unable to perform double gaussian fit "<<endl;
    // func.SetParameters(0);
    hist->Fit(&func, "Q0LR");
  }
  
  return;
}

/*--------------------------------------------------------------------*/
Double_t fDLine(Double_t *x, Double_t *par)
/*--------------------------------------------------------------------*/
{
  if (x[0] < _boundSx && x[0] > _boundDx) {
    TF1::RejectPoint();
    return 0;
  }
  return par[0];
}

/*--------------------------------------------------------------------*/
Double_t fULine(Double_t *x, Double_t *par)
/*--------------------------------------------------------------------*/
{
  if (x[0] >= _boundSx && x[0] <= _boundDx) {
    TF1::RejectPoint();
    return 0;
  }
  return par[0];
}

/*--------------------------------------------------------------------*/
void FitULine(TH1 *hist)
/*--------------------------------------------------------------------*/
{ 
  // define fitting function
  TF1 func1("lineUp",fULine,_boundMin,_boundMax,1);
  //TF1 func1("lineUp","pol0",-0.5,11.5);
  
  if (0 == hist->Fit(&func1,"QR")) {
    if (hist->GetFunction(func1.GetName())) { // Take care that it is later on drawn:
      hist->GetFunction(func1.GetName())->ResetBit(TF1::kNotDraw);
    }
    cout<<"fit Up done!"<<endl;
  }
  
}

/*--------------------------------------------------------------------*/
void FitDLine(TH1 *hist)
/*--------------------------------------------------------------------*/
{
  // define fitting function
  // TF1 func1("lineDown",fDLine,-0.5,11.5,1);
  
  TF1 func2("lineDown","pol0",_boundSx,_boundDx);
  func2.SetRange(_boundSx,_boundDx);
  
  if (0 == hist->Fit(&func2,"QR")) {
    if (hist->GetFunction(func2.GetName())) { // Take care that it is later on drawn:
      hist->GetFunction(func2.GetName())->ResetBit(TF1::kNotDraw);
    }
    cout<<"fit Down done!"<<endl;
  } 
}

/*--------------------------------------------------------------------*/
void MakeNiceTF1Style(TF1 *f1,int color)
/*--------------------------------------------------------------------*/
{
  f1->SetLineColor(color);
  f1->SetLineWidth(3);
  f1->SetLineStyle(2);
}

/*--------------------------------------------------------------------*/
void MakeNiceHistoStyle(TH1 *hist)
/*--------------------------------------------------------------------*/
{
  
  //hist->SetTitleSize(0.09); 
  hist->GetYaxis()->SetTitleOffset(1.2);
  hist->GetXaxis()->SetTitleOffset(0.9);
  hist->GetYaxis()->SetTitleSize(0.08);
  hist->GetXaxis()->SetTitleSize(0.07);
  hist->GetXaxis()->SetTitleFont(42); 
  hist->GetYaxis()->SetTitleFont(42);  
  hist->GetXaxis()->SetLabelSize(0.05); 
  hist->GetYaxis()->SetLabelSize(0.05);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  hist->SetMarkerSize(1.2);
  
}
/*--------------------------------------------------------------------*/
void MakeNiceTrendPlotStyle(TH1 *hist)
/*--------------------------------------------------------------------*/
{ 
  hist->SetStats(kFALSE);  
  hist->SetLineWidth(2);
  hist->GetXaxis()->CenterTitle(true);
  hist->GetYaxis()->CenterTitle(true);
  hist->GetXaxis()->SetTitleFont(42); 
  hist->GetYaxis()->SetTitleFont(42);  
  hist->GetXaxis()->SetTitleSize(0.06);
  hist->GetYaxis()->SetTitleSize(0.06);
  hist->GetXaxis()->SetTitleOffset(1.0);
  hist->GetYaxis()->SetTitleOffset(1.0);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelSize(.06);
  hist->GetXaxis()->SetLabelSize(.05);
  hist->SetMarkerSize(1.3);
}

//
// Sinusoidal functions to plot
//
/*--------------------------------------------------------------------*/
Double_t simplesine(Double_t *x, Double_t *par)
/*--------------------------------------------------------------------*/
{
  Float_t xx = x[0];
  Double_t fitval =-par[1]*cos((TMath::Pi()/_divs)*(xx-_divs))+par[0]*sin((TMath::Pi()/_divs)*(xx-_divs));
  return fitval;  
}

/*--------------------------------------------------------------------*/
Double_t splitfunc(Double_t *x, Double_t *par)
/*--------------------------------------------------------------------*/
{
  Double_t xx = x[0];
  Double_t dxhs = -par[1]*cos((TMath::Pi()/_divs)*(xx-_divs))+par[0]*sin((TMath::Pi()/_divs)*(xx-_divs)) + par[2]*(1 - sin((TMath::Pi()/_divs)*(xx-_divs)));
  Double_t sxhs = -par[1]*cos((TMath::Pi()/_divs)*(xx-_divs))+par[0]*sin((TMath::Pi()/_divs)*(xx-_divs)) - par[3]*(1 - sin((TMath::Pi()/_divs)*(xx-_divs)));
						   
  if (xx > _boundSx && xx < _boundDx) {
    return dxhs;
  } else if (xx> _boundMin && xx < _boundSx) {
    return sxhs;
  } else if (xx == _boundSx && xx == _boundDx) {
    return par[0];
  } else if (xx > _boundDx && xx < _boundMax)
    return sxhs;
  else {
    std::cout<<"shouldn't ever be the case"<<std::endl;
    return 0;
  }
}

/*--------------------------------------------------------------------*/
Double_t splitfunconly(Double_t *x, Double_t *par)
/*--------------------------------------------------------------------*/
{
  Double_t xx = x[0];
  Double_t dxhs =  par[0]*(1 - sin((TMath::Pi()/_divs)*(xx-_divs)));
  Double_t sxhs = -par[1]*(1 - sin((TMath::Pi()/_divs)*(xx-_divs)));
						   
  if (xx >= 2.50 && xx <= 8.50) {
    return dxhs;
  } else {
    return sxhs;
  }
}

/*--------------------------------------------------------------------*/
Double_t myCosine(Double_t *x, Double_t *par)
/*--------------------------------------------------------------------*/
{
  Float_t xx = x[0];
  Double_t f = par[0]*cos((TMath::Pi()/_divs)*(xx-_divs) + par[2]);
  return f;
}

/*--------------------------------------------------------------------*/
void doubleGausFitResiduals(TH1F *hist)
/*--------------------------------------------------------------------*/
{

  //TH1F *hist2 = (TH1F*)hist->Clone("h2");
  //Use TSpectrum to find the peak candidates
  TSpectrum *s = new TSpectrum(2);
  Int_t nfound = s->Search(hist,1,"new");
  printf("Found %d candidate peaks to fit \n",nfound);
  Float_t *xpeaks = s->GetPositionX();
  
  float mean   = hist->GetMean();
  float sigma  = hist->GetRMS();
  float x_peak = xpeaks[0];
  float fwhm   = calcFWHM(hist);

  std::cout<<" mean: "<<mean 
	   <<" x_peak: "<<x_peak 
	   <<" sigma: "<<sigma
	   <<" fwhm: "<<fwhm<<std::endl; 
  
  float min1_ = x_peak-fwhm; 
  float max1_ = x_peak+fwhm;
 		      
  float min2_ = x_peak-2*fwhm;
  float max2_ = x_peak+2*fwhm;
   
  Double_t par[6];
  TF1 *g1    = new TF1("g1","gaus",min1_,max1_);
  TF1 *g2    = new TF1("g2","gaus",min2_,max2_);
  TF1 *func = new TF1("tmp","gaus(0)+gaus(3)",min2_,max2_);
  hist->Fit(g1,"QNR");
  hist->Fit(g2,"QNR+");

  g1->GetParameters(&par[0]);
  g2->GetParameters(&par[3]);
  func->SetParameters(par);
  // func->FixParameter(1,x_peak);
  // func->FixParameter(3,x_peak); 
  
  hist->Fit(func,"Q0LR+");
  
  hist->Draw();
  func->Draw("same");
}


/*--------------------------------------------------------------------*/
Float_t calcFWHM(TH1F *hist)
/*--------------------------------------------------------------------*/
{

  Float_t FWHM(999.);

  Int_t bin_max = hist->GetMaximumBin();
  Int_t the_max = hist->GetMaximum();
  Int_t halfmax_bin_sx(-999999);
  Int_t halfmax_bin_dx(999999);  
  
  for(Int_t i=1; i<bin_max; i++){
    if(hist->GetBinContent(i) > the_max/2. ){
      halfmax_bin_sx = i-1;
      break;
    }  
  }
  
  for(Int_t j=bin_max; j<hist->GetNbinsX(); j++){
    //std::cout<<"bin: "<<j<<" bincontent:"<<hist->GetBinContent(j)<<std::endl;
    if(hist->GetBinContent(j) < the_max/2. ){
      halfmax_bin_dx = j+1;
      break;
    }  
  } 
  
  // std::cout<<"themax: "<<the_max<<" binmax: "<<bin_max<<" halfmax_bin_sx: "<< halfmax_bin_sx << "  halfmax_bin_dx: "<< halfmax_bin_dx<<std::endl;

  FWHM = hist->GetBinCenter(halfmax_bin_dx) - hist->GetBinCenter(halfmax_bin_sx);
  if(FWHM>3000.){
    std::cout<<"hist name:"<<hist->GetName()<<std::endl;
    std::cout<<"themax: "<<the_max<<" binmax: "<<bin_max<<" halfmax_bin_sx: "<< halfmax_bin_sx << "  halfmax_bin_dx: "<< halfmax_bin_dx<<std::endl;
  }
  // std::cout<<"FWHM="<<FWHM<<std::endl;
  return FWHM;
  
}
