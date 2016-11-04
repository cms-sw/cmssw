#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <Riostream.h>
#include "TFile.h"
#include "TPaveStats.h"
#include "TROOT.h"
#include "TList.h"
#include "TNtuple.h"
#include "TTree.h"
#include "TH1.h"
#include "TArrow.h"
#include "TH2.h"
#include "THStack.h"
#include "TStyle.h"
#include "TLegendEntry.h"
#include "TPaveText.h"
#include "TCut.h"
#include "TLegend.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "TMath.h"
#include "TVectorD.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TAxis.h"
#include "TGaxis.h"
#include "TROOT.h"
#include "TObjArray.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "TMinuit.h"
#include "TString.h"
#include "TMath.h"
#include <TDatime.h>
#include <TSpectrum.h>
#include <TSystem.h>
#include <TStopwatch.h>
#include "Alignment/OfflineValidation/plugins/TkAlStyle.cc" 
#include "CMS_lumi.C"


/*
  This is an auxilliary struct used to handle the plot limits
 */

struct Limits {

  // initializers list

  Limits() : _m_dxyPhiMax(40.),     _m_dzPhiMax(40.),       
	     _m_dxyEtaMax(40.),     _m_dzEtaMax(40.),       
	     _m_dxyPhiNormMax(0.5), _m_dzPhiNormMax(0.5),   
	     _m_dxyEtaNormMax(0.5), _m_dzEtaNormMax(0.5),   
	     _w_dxyPhiMax(120.),    _w_dzPhiMax(180.),      
	     _w_dxyEtaMax(120.),    _w_dzEtaMax(1000.),       
	     _w_dxyPhiNormMax(1.8), _w_dzPhiNormMax(1.8),   
	     _w_dxyEtaNormMax(1.8), _w_dzEtaNormMax(1.8) {}   

  // getter methods

  std::pair <float,float> get_dxyPhiMax() const {
    std::pair <float , float> res(_m_dxyPhiMax,_w_dxyPhiMax);
    return res;
  }
  
  std::pair <float,float> get_dzPhiMax() const {
    std::pair <float , float> res(_m_dzPhiMax,_w_dzPhiMax);
    return res;
  }
  
  std::pair <float,float> get_dxyEtaMax() const {
   std::pair <float , float> res(_m_dxyEtaMax,_w_dxyEtaMax);
   return res;
 }
  
  std::pair <float,float> get_dzEtaMax() const {
    std::pair <float , float> res(_m_dzEtaMax,_w_dzEtaMax);
    return res;
  }
  
 std::pair <float,float> get_dxyPhiNormMax() const {
   std::pair <float , float> res(_m_dxyPhiNormMax,_w_dxyPhiNormMax);
   return res;
 }
  
  std::pair <float,float> get_dzPhiNormMax() const {
    std::pair <float , float> res(_m_dzPhiNormMax,_w_dzPhiNormMax);
    return res;
  }
  
  std::pair <float,float> get_dxyEtaNormMax() const {
    std::pair <float , float> res(_m_dxyEtaNormMax,_w_dxyEtaNormMax);
    return res;
  }
  
  std::pair <float,float> get_dzEtaNormMax() const {
    std::pair <float , float> res(_m_dzEtaNormMax,_w_dzEtaNormMax);
    return res;
  }

  // initializes to different values, if needed

  void init(float m_dxyPhiMax,float m_dzPhiMax,float m_dxyEtaMax,float m_dzEtaMax,        		  
	    float m_dxyPhiNormMax,float m_dzPhiNormMax,float m_dxyEtaNormMax,float m_dzEtaNormMax,    
	    float w_dxyPhiMax,float w_dzPhiMax,float w_dxyEtaMax,float w_dzEtaMax,	  		  
	    float w_dxyPhiNormMax,float w_dzPhiNormMax,float w_dxyEtaNormMax,float w_dzEtaNormMax){
    
    _m_dxyPhiMax     = m_dxyPhiMax;       
    _m_dzPhiMax      = m_dzPhiMax;        
    _m_dxyEtaMax     = m_dxyEtaMax;       
    _m_dzEtaMax      = m_dzEtaMax;                                 
    _m_dxyPhiNormMax = m_dxyPhiNormMax; 
    _m_dzPhiNormMax  = m_dzPhiNormMax; 
    _m_dxyEtaNormMax = m_dxyEtaNormMax; 
    _m_dzEtaNormMax  = m_dzEtaNormMax;     
    _w_dxyPhiMax     = w_dxyPhiMax; 
    _w_dzPhiMax      = w_dzPhiMax; 
    _w_dxyEtaMax     = w_dxyEtaMax; 
    _w_dzEtaMax      = w_dzEtaMax;
    _w_dxyPhiNormMax = w_dxyPhiNormMax; 
    _w_dzPhiNormMax  = w_dzPhiNormMax; 
    _w_dxyEtaNormMax = w_dxyEtaNormMax; 
    _w_dzEtaNormMax  = w_dzEtaNormMax;    
    
  }

  void printAll(){
    std::cout<<"======================================================"<<std::endl;
    std::cout<<"  The y-axis ranges on the plots will be the following:      "<<std::endl;
    std::cout<<"  m_dxyPhiMax:" <<      _m_dxyPhiMax<< std::endl;  
    std::cout<<"  m_dzPhiMax:"  <<      _m_dzPhiMax<< std::endl;  
    std::cout<<"  m_dxyEtaMax:" <<      _m_dxyEtaMax<< std::endl;  
    std::cout<<"  m_dzEtaMax:"  <<      _m_dzEtaMax<< std::endl;  
    
    std::cout<<"  m_dxyPhiNormMax:" <<  _m_dxyPhiNormMax<< std::endl; 
    std::cout<<"  m_dzPhiNormMax:"  <<  _m_dzPhiNormMax<< std::endl; 
    std::cout<<"  m_dxyEtaNormMax:" <<  _m_dxyEtaNormMax<< std::endl; 
    std::cout<<"  m_dzEtaNormMax:"  <<  _m_dzEtaNormMax<< std::endl; 
    
    std::cout<<"  w_dxyPhiMax:" <<      _w_dxyPhiMax<< std::endl; 
    std::cout<<"  w_dzPhiMax:"  <<      _w_dzPhiMax<< std::endl; 
    std::cout<<"  w_dxyEtaMax:" <<      _w_dxyEtaMax<< std::endl; 
    std::cout<<"  w_dzEtaMax:"  <<      _w_dzEtaMax<< std::endl;
    
    std::cout<<"  w_dxyPhiNormMax:" <<  _w_dxyPhiNormMax<< std::endl; 
    std::cout<<"  w_dzPhiNormMax:"  <<  _w_dzPhiNormMax<< std::endl; 
    std::cout<<"  w_dxyEtaNormMax:" <<  _w_dxyEtaNormMax<< std::endl; 
    std::cout<<"  w_dzEtaNormMax:"  <<  _w_dzEtaNormMax<< std::endl; 
 
    std::cout<<"======================================================"<<std::endl;
  }

private:
  float _m_dxyPhiMax;    
  float _m_dzPhiMax;    
  float _m_dxyEtaMax;    
  float _m_dzEtaMax;                            
  float _m_dxyPhiNormMax;   
  float _m_dzPhiNormMax;   
  float _m_dxyEtaNormMax;   
  float _m_dzEtaNormMax;
                           
  float _w_dxyPhiMax;   
  float _w_dzPhiMax;   
  float _w_dxyEtaMax;   
  float _w_dzEtaMax;                          
  float _w_dxyPhiNormMax;   
  float _w_dzPhiNormMax;   
  float _w_dxyEtaNormMax;   
  float _w_dzEtaNormMax;    
  
};

Limits* thePlotLimits = new Limits();

#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array[0])))

void arrangeCanvas(TCanvas *canv,TH1F* meanplots[100],TH1F* widthplots[100],Int_t nFiles,TString LegLabels[10],TString theDate="bogus",bool onlyBias=false,bool setAutoLimits=true);
void arrangeCanvas2D(TCanvas *canv,TH2F* meanmaps[100],TH2F* widthmaps[100],Int_t nFiles,TString LegLabels[10],TString theDate="bogus");
void arrangeFitCanvas(TCanvas *canv,TH1F* meanplots[100],Int_t nFiles, TString LegLabels[10],TString theDate="bogus");

void arrangeBiasCanvas(TCanvas *canv,TH1F* dxyPhiMeanTrend[100],TH1F* dzPhiMeanTrend[100],TH1F* dxyEtaMeanTrend[100],TH1F* dzEtaMeanTrend[100],Int_t nFiles, TString LegLabels[10],TString theDate="bogus",bool setAutoLimits=true);

std::pair<Double_t,Double_t> getMedian(TH1F *histo);
std::pair<Double_t,Double_t> getMAD(TH1F *histo);

std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > fitResiduals(TH1 *hist);

Double_t DoubleSidedCB(double* x, double* par);
std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > fitResidualsCB(TH1 *hist);

Double_t tp0Fit( Double_t *x, Double_t *par5 );
std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > fitStudentTResiduals(TH1 *hist);

void FillTrendPlot(TH1F* trendPlot, TH1F* residualsPlot[100], TString fitPar_, TString var_,Int_t nbins);
void FillMap(TH2F* trendMap, TH1F* residualsMapPlot[48][48], TString fitPar_);

void MakeNiceTrendPlotStyle(TH1 *hist,Int_t color);
void MakeNicePlotStyle(TH1 *hist);
void MakeNiceMapStyle(TH2 *hist);
void MakeNiceTF1Style(TF1 *f1,Int_t color);

void FitPVResiduals(TString namesandlabels,bool stdres=true,bool do2DMaps=false,TString theDate="bogus",bool setAutoLimits=true);
TH1F* DrawZero(TH1F *hist,Int_t nbins,Double_t lowedge,Double_t highedge,Int_t iter);
void makeNewXAxis (TH1F *h);
void makeNewPairOfAxes (TH2F *h);

// ancillary fitting functions
Double_t fULine(Double_t *x, Double_t *par);
Double_t fDLine(Double_t *x, Double_t *par);
void FitULine(TH1 *hist);
void FitDLine(TH1 *hist);

std::pair<Double_t,Double_t> getTheRangeUser(TH1F* thePlot,Limits* thePlotLimits);

void setStyle();

// global variables

ofstream outfile("FittedDeltaZ.txt");
Int_t my_colors[10]={kBlack,kRed,kBlue,kMagenta,kBlack,kRed,kBlue,kGreen,kOrange,kViolet};

const Int_t nBins_  = 48;
Float_t _boundMin   = -0.5;
Float_t _boundSx    = (nBins_/4.)-0.5;
Float_t _boundDx    = 3*(nBins_/4.)-0.5;
Float_t _boundMax   = nBins_-0.5;

//*************************************************************
void FitPVResiduals(TString namesandlabels,bool stdres,bool do2DMaps,TString theDate,bool setAutoLimits){
//*************************************************************
  
  TStopwatch timer; 	 
  timer.Start();

  if(!setAutoLimits){
    std::cout<<" FitPVResiduals: Overriding autolimits!"<<std::endl;
    thePlotLimits->printAll();
  } else {
    std::cout<<" FitPVResiduals: plot axis range will be automatically adjusted"<<std::endl;
  }
    

  TkAlStyle::set(PRELIMINARY);	// set publication status

  Int_t colors[10]={0,1,2,3,4,5,6,7,8,9};
  setStyle();

  TList *FileList  = new TList();
  TList *LabelList = new TList();
  
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

  const Int_t nFiles_ = FileList->GetSize();
  TString LegLabels[10];  
  TFile *fins[nFiles_]; 

  for(Int_t j=0; j < nFiles_; j++) {
    
    // Retrieve files
    fins[j] = (TFile*)FileList->At(j);    
 
    // Retrieve labels
    TObjString* legend = (TObjString*)LabelList->At(j);
    LegLabels[j] = legend->String();
    LegLabels[j].ReplaceAll("_"," ");
    cout<<"FitPVResiduals::FitPVResiduals(): label["<<j<<"]"<<LegLabels[j]<<endl;
    
  }

  // already used in the global variables
  //const Int_t nBins_ =24;
  
  // dca absolute residuals
  TH1F* dxyPhiResiduals[nFiles_][nBins_];
  TH1F* dxyEtaResiduals[nFiles_][nBins_];
  				        
  TH1F* dzPhiResiduals[nFiles_][nBins_];
  TH1F* dzEtaResiduals[nFiles_][nBins_];

  // dca normalized residuals
  TH1F* dxyNormPhiResiduals[nFiles_][nBins_];
  TH1F* dxyNormEtaResiduals[nFiles_][nBins_];
  				        
  TH1F* dzNormPhiResiduals[nFiles_][nBins_];
  TH1F* dzNormEtaResiduals[nFiles_][nBins_];
  
  // double-differential residuals
  TH1F* dxyMapResiduals[nFiles_][nBins_][nBins_]; 
  TH1F* dzMapResiduals[nFiles_][nBins_][nBins_];     
  
  TH1F* dxyNormMapResiduals[nFiles_][nBins_][nBins_];
  TH1F* dzNormMapResiduals[nFiles_][nBins_][nBins_]; 
  
  for(Int_t i=0;i<nFiles_;i++){
    for(Int_t j=0;j<nBins_;j++){
      
      if(stdres){
	// DCA absolute residuals
	dxyPhiResiduals[i][j] = (TH1F*)fins[i]->Get(Form("PVValidation/Abs_Transv_Phi_Residuals/histo_dxy_phi_plot%i",j));
	dxyEtaResiduals[i][j] = (TH1F*)fins[i]->Get(Form("PVValidation/Abs_Transv_Eta_Residuals/histo_dxy_eta_plot%i",j));
	dzPhiResiduals[i][j]  = (TH1F*)fins[i]->Get(Form("PVValidation/Abs_Long_Phi_Residuals/histo_dz_phi_plot%i",j));
	dzEtaResiduals[i][j]  = (TH1F*)fins[i]->Get(Form("PVValidation/Abs_Long_Eta_Residuals/histo_dz_eta_plot%i",j));
	
	// DCA normalized residuals
	dxyNormPhiResiduals[i][j] = (TH1F*)fins[i]->Get(Form("PVValidation/Norm_Transv_Phi_Residuals/histo_norm_dxy_phi_plot%i",j));
	dxyNormEtaResiduals[i][j] = (TH1F*)fins[i]->Get(Form("PVValidation/Norm_Transv_Eta_Residuals/histo_norm_dxy_eta_plot%i",j));
	dzNormPhiResiduals[i][j]  = (TH1F*)fins[i]->Get(Form("PVValidation/Norm_Long_Phi_Residuals/histo_norm_dz_phi_plot%i",j));
	dzNormEtaResiduals[i][j]  = (TH1F*)fins[i]->Get(Form("PVValidation/Norm_Long_Eta_Residuals/histo_norm_dz_eta_plot%i",j));

	// double differential residuals
	
	for(Int_t k=0;k<nBins_;k++){
	  
	  // absolute residuals
	  dxyMapResiduals[i][j][k] = (TH1F*)fins[i]->Get(Form("PVValidation/Abs_DoubleDiffResiduals/histo_dxy_eta_plot%i_phi_plot%i",j,k));	  
	  dzMapResiduals[i][j][k]  = (TH1F*)fins[i]->Get(Form("PVValidation/Abs_DoubleDiffResiduals/histo_dz_eta_plot%i_phi_plot%i",j,k));  
	  
	  // normalized residuals
	  dxyNormMapResiduals[i][j][k] = (TH1F*)fins[i]->Get(Form("PVValidation/Norm_DoubleDiffResiduals/histo_norm_dxy_eta_plot%i_phi_plot%i",j,k));  
	  dzNormMapResiduals[i][j][k]  = (TH1F*)fins[i]->Get(Form("PVValidation/Norm_DoubleDiffResiduals/histo_norm_dz_eta_plot%i_phi_plot%i",j,k));

	}

      } else {

	// DCA absolute residuals
	dxyPhiResiduals[i][j] = (TH1F*)fins[i]->Get(Form("PVValidation/Abs_Transv_Phi_Residuals/histo_IP2D_phi_plot%i",j));
	dxyEtaResiduals[i][j] = (TH1F*)fins[i]->Get(Form("PVValidation/Abs_Transv_Eta_Residuals/histo_IP2D_eta_plot%i",j));
	dzPhiResiduals[i][j]  = (TH1F*)fins[i]->Get(Form("PVValidation/Abs_Long_Phi_Residuals/histo_resz_phi_plot%i",j));
	dzEtaResiduals[i][j]  = (TH1F*)fins[i]->Get(Form("PVValidation/Abs_Long_Eta_Residuals/histo_resz_eta_plot%i",j));
	
	// DCA normalized residuals
	dxyNormPhiResiduals[i][j] = (TH1F*)fins[i]->Get(Form("PVValidation/Norm_Transv_Phi_Residuals/histo_norm_IP2D_phi_plot%i",j));
	dxyNormEtaResiduals[i][j] = (TH1F*)fins[i]->Get(Form("PVValidation/Norm_Transv_Eta_Residuals/histo_norm_IP2D_eta_plot%i",j));
	dzNormPhiResiduals[i][j]  = (TH1F*)fins[i]->Get(Form("PVValidation/Norm_Long_Phi_Residuals/histo_norm_resz_phi_plot%i",j));
	dzNormEtaResiduals[i][j]  = (TH1F*)fins[i]->Get(Form("PVValidation/Norm_Long_Eta_Residuals/histo_norm_resz_eta_plot%i",j));


	// double differential residuals
      
	for(Int_t k=0;k<nBins_;k++){

	  // absolute residuals
	  dxyMapResiduals[i][j][k] = (TH1F*)fins[i]->Get(Form("PVValidation/Abs_DoubleDiffResiduals/histo_dxy_eta_plot%i_phi_plot%i",j,k));
	  dzMapResiduals[i][j][k]  = (TH1F*)fins[i]->Get(Form("PVValidation/Abs_DoubleDiffResiduals/histo_dz_eta_plot%i_phi_plot%i",j,k));  
	  
	  // normalized residuals
	  dxyNormMapResiduals[i][j][k] = (TH1F*)fins[i]->Get(Form("PVValidation/Norm_DoubleDiffResiduals/histo_norm_dxy_eta_plot%i_phi_plot%i",j,k));  
	  dzNormMapResiduals[i][j][k]  = (TH1F*)fins[i]->Get(Form("PVValidation/Norm_DoubleDiffResiduals/histo_norm_dz_eta_plot%i_phi_plot%i",j,k));

	}
      }
    }
  }
 
  Double_t highedge=nBins_-0.5;
  Double_t lowedge=-0.5;
  
  // DCA absolute

  TH1F* dxyPhiMeanTrend[nFiles_];  
  TH1F* dxyPhiWidthTrend[nFiles_]; 
  TH1F* dzPhiMeanTrend[nFiles_];   
  TH1F* dzPhiWidthTrend[nFiles_];  
  		       	 
  TH1F* dxyEtaMeanTrend[nFiles_];  
  TH1F* dxyEtaWidthTrend[nFiles_]; 
  TH1F* dzEtaMeanTrend[nFiles_];   
  TH1F* dzEtaWidthTrend[nFiles_];  

  // DCA normalized

  TH1F* dxyNormPhiMeanTrend[nFiles_];  
  TH1F* dxyNormPhiWidthTrend[nFiles_]; 
  TH1F* dzNormPhiMeanTrend[nFiles_];   
  TH1F* dzNormPhiWidthTrend[nFiles_];  
  		       	 
  TH1F* dxyNormEtaMeanTrend[nFiles_];  
  TH1F* dxyNormEtaWidthTrend[nFiles_]; 
  TH1F* dzNormEtaMeanTrend[nFiles_];   
  TH1F* dzNormEtaWidthTrend[nFiles_];  

  // 2D maps

  // bias
  TH2F* dxyMeanMap[nFiles_];  
  TH2F* dzMeanMap[nFiles_];   
  TH2F* dxyNormMeanMap[nFiles_];  
  TH2F* dzNormMeanMap[nFiles_];
   
  // width
  TH2F* dxyWidthMap[nFiles_]; 
  TH2F* dzWidthMap[nFiles_];  		
  TH2F* dxyNormWidthMap[nFiles_]; 
  TH2F* dzNormWidthMap[nFiles_];


  for(Int_t i=0;i<nFiles_;i++){

    // DCA trend plots

    dxyPhiMeanTrend[i]  = new TH1F(Form("means_dxy_phi_%i",i),"#LT d_{xy} #GT vs #phi sector;track #phi [rad];#LT d_{xy} #GT [#mum]",nBins_,lowedge,highedge); 
    dxyPhiWidthTrend[i] = new TH1F(Form("widths_dxy_phi_%i",i),"#sigma(d_{xy}) vs #phi sector;track #phi [rad];#sigma(d_{xy}) [#mum]",nBins_,lowedge,highedge);
    dzPhiMeanTrend[i]   = new TH1F(Form("means_dz_phi_%i",i),"#LT d_{z} #GT vs #phi sector;track #phi [rad];#LT d_{z} #GT [#mum]",nBins_,lowedge,highedge); 
    dzPhiWidthTrend[i]  = new TH1F(Form("widths_dz_phi_%i",i),"#sigma(d_{z}) vs #phi sector;track #phi [rad];#sigma(d_{z}) [#mum]",nBins_,lowedge,highedge);
    
    dxyEtaMeanTrend[i]  = new TH1F(Form("means_dxy_eta_%i",i),"#LT d_{xy} #GT vs #eta sector;track #eta;#LT d_{xy} #GT [#mum]",nBins_,lowedge,highedge);
    dxyEtaWidthTrend[i] = new TH1F(Form("widths_dxy_eta_%i",i),"#sigma(d_{xy}) vs #eta sector;track #eta;#sigma(d_{xy}) [#mum]",nBins_,lowedge,highedge);
    dzEtaMeanTrend[i]   = new TH1F(Form("means_dz_eta_%i",i),"#LT d_{z} #GT vs #eta sector;track #eta;#LT d_{z} #GT [#mum]",nBins_,lowedge,highedge); 
    dzEtaWidthTrend[i]  = new TH1F(Form("widths_dz_eta_%i",i),"#sigma(d_{xy}) vs #eta sector;track #eta;#sigma(d_{z}) [#mum]",nBins_,lowedge,highedge);

    dxyNormPhiMeanTrend[i] = new TH1F(Form("means_dxyNorm_phi_%i",i),"#LT d_{xy}/#sigma_{d_{xy}} #GT vs #phi sector;track #phi [rad];#LT d_{xy}/#sigma_{d_{xy}} #GT [#mum]",nBins_,lowedge,highedge); 
    dxyNormPhiWidthTrend[i]= new TH1F(Form("widths_dxyNorm_phi_%i",i),"#sigma(d_{xy}/#sigma_{d_{xy}}) vs #phi sector;track #phi [rad];#sigma(d_{xy}/#sigma_{d_{xy}}) [#mum]",nBins_,lowedge,highedge);
    dzNormPhiMeanTrend[i]  = new TH1F(Form("means_dzNorm_phi_%i",i),"#LT d_{z}/#sigma_{d_{z}} #GT vs #phi sector;track #phi [rad];#LT d_{z}/#sigma_{d_{z}} #GT [#mum]",nBins_,lowedge,highedge); 
    dzNormPhiWidthTrend[i] = new TH1F(Form("widths_dzNorm_phi_%i",i),"#sigma(d_{z}/#sigma_{d_{z}}) vs #phi sector;track #phi [rad];#sigma(d_{z}/#sigma_{d_{z}}) [#mum]",nBins_,lowedge,highedge);
    
    dxyNormEtaMeanTrend[i] = new TH1F(Form("means_dxyNorm_eta_%i",i),"#LT d_{xy}/#sigma_{d_{xy}} #GT vs #eta sector;track #eta;#LT d_{xy}/#sigma_{d_{xy}} #GT [#mum]",nBins_,lowedge,highedge);
    dxyNormEtaWidthTrend[i]= new TH1F(Form("widths_dxyNorm_eta_%i",i),"#sigma(d_{xy}/#sigma_{d_{xy}}) vs #eta sector;track #eta;#sigma(d_{xy}/#sigma_{d_{xy}}) [#mum]",nBins_,lowedge,highedge);
    dzNormEtaMeanTrend[i]  = new TH1F(Form("means_dzNorm_eta_%i",i),"#LT d_{z}/#sigma_{d_{z}} #GT vs #eta sector;track #eta;#LT d_{z}/#sigma_{d_{z}} #GT [#mum]",nBins_,lowedge,highedge); 
    dzNormEtaWidthTrend[i] = new TH1F(Form("widths_dzNorm_eta_%i",i),"#sigma(d_{z}/#sigma_{d_{z}}) vs #eta sector;track #eta;#sigma(d_{z}/#sigma_{d_{z}}) [#mum]",nBins_,lowedge,highedge);
   

    // 2D maps
    dxyMeanMap[i]      = new TH2F(Form("means_dxy_map_%i",i), "#LT d_{xy} #GT map;track #eta;track #phi [rad];#LT d_{xy} #GT [#mum]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);		       
    dzMeanMap[i]       = new TH2F(Form("means_dz_map_%i",i),"#LT d_{z} #GT map;track #eta;track #phi [rad];#LT d_{z} #GT [#mum]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);  		       
    dxyNormMeanMap[i]  = new TH2F(Form("norm_means_dxy_map_%i",i),"#LT d_{xy}/#sigma_{d_{xy}} #GT map;track #eta;track #phi [rad];#LT d_{xy}/#sigma_{d_{xy}} #GT",nBins_,lowedge,highedge,nBins_,lowedge,highedge);  
    dzNormMeanMap[i]   = new TH2F(Form("norm_means_dz_map_%i",i),"#LT d_{z}/#sigma_{d_{z}} #GT map;track #eta;track #phi[rad];#LT d_{xy}/#sigma_{d_{z}} #GT",nBins_,lowedge,highedge,nBins_,lowedge,highedge);    
 
    dxyWidthMap[i]     = new TH2F(Form("widths_dxy_map_%i",i),"#sigma_{d_{xy}} map;track #eta;track #phi [rad];#sigma(d_{xy}) [#mum]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);		       
    dzWidthMap[i]      = new TH2F(Form("widths_dz_map_%i",i),"#sigma_{d_{z}} map;track #eta;track #phi [rad];#sigma(d_{z}) [#mum]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);  		       
    dxyNormWidthMap[i] = new TH2F(Form("norm_widths_dxy_map_%i",i),"width(d_{xy}/#sigma_{d_{xy}}) map;track #eta;track #phi[rad];#sigma(d_{xy}/#sigma_{d_{xy}})",nBins_,lowedge,highedge,nBins_,lowedge,highedge);  
    dzNormWidthMap[i]  = new TH2F(Form("norm_widths_dz_map_%i",i),"width(d_{z}/#sigma_{d_{z}}) map;track #eta;track #phi [rad];#sigma(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge,nBins_,lowedge,highedge); 

    // DCA absolute

    FillTrendPlot(dxyPhiMeanTrend[i] ,dxyPhiResiduals[i],"mean","phi",nBins_);  
    FillTrendPlot(dxyPhiWidthTrend[i],dxyPhiResiduals[i],"width","phi",nBins_);
    FillTrendPlot(dzPhiMeanTrend[i]  ,dzPhiResiduals[i] ,"mean","phi",nBins_);   
    FillTrendPlot(dzPhiWidthTrend[i] ,dzPhiResiduals[i] ,"width","phi",nBins_);  
    
    FillTrendPlot(dxyEtaMeanTrend[i] ,dxyEtaResiduals[i],"mean","eta",nBins_); 
    FillTrendPlot(dxyEtaWidthTrend[i],dxyEtaResiduals[i],"width","eta",nBins_);
    FillTrendPlot(dzEtaMeanTrend[i]  ,dzEtaResiduals[i] ,"mean","eta",nBins_); 
    FillTrendPlot(dzEtaWidthTrend[i] ,dzEtaResiduals[i] ,"width","eta",nBins_);

    MakeNiceTrendPlotStyle(dxyPhiMeanTrend[i],colors[i]);
    MakeNiceTrendPlotStyle(dxyPhiWidthTrend[i],colors[i]);
    MakeNiceTrendPlotStyle(dzPhiMeanTrend[i],colors[i]);
    MakeNiceTrendPlotStyle(dzPhiWidthTrend[i],colors[i]);
  
    MakeNiceTrendPlotStyle(dxyEtaMeanTrend[i],colors[i]);
    MakeNiceTrendPlotStyle(dxyEtaWidthTrend[i],colors[i]);
    MakeNiceTrendPlotStyle(dzEtaMeanTrend[i],colors[i]);
    MakeNiceTrendPlotStyle(dzEtaWidthTrend[i],colors[i]);
    
    // DCA normalized

    FillTrendPlot(dxyNormPhiMeanTrend[i] ,dxyNormPhiResiduals[i],"mean","phi",nBins_);  
    FillTrendPlot(dxyNormPhiWidthTrend[i],dxyNormPhiResiduals[i],"width","phi",nBins_);
    FillTrendPlot(dzNormPhiMeanTrend[i]  ,dzNormPhiResiduals[i] ,"mean","phi",nBins_);   
    FillTrendPlot(dzNormPhiWidthTrend[i] ,dzNormPhiResiduals[i] ,"width","phi",nBins_);  
    
    FillTrendPlot(dxyNormEtaMeanTrend[i] ,dxyNormEtaResiduals[i],"mean","eta",nBins_); 
    FillTrendPlot(dxyNormEtaWidthTrend[i],dxyNormEtaResiduals[i],"width","eta",nBins_);
    FillTrendPlot(dzNormEtaMeanTrend[i]  ,dzNormEtaResiduals[i] ,"mean","eta",nBins_); 
    FillTrendPlot(dzNormEtaWidthTrend[i] ,dzNormEtaResiduals[i] ,"width","eta",nBins_);

    MakeNiceTrendPlotStyle(dxyNormPhiMeanTrend[i],colors[i]);
    MakeNiceTrendPlotStyle(dxyNormPhiWidthTrend[i],colors[i]);
    MakeNiceTrendPlotStyle(dzNormPhiMeanTrend[i],colors[i]);
    MakeNiceTrendPlotStyle(dzNormPhiWidthTrend[i],colors[i]);
  
    MakeNiceTrendPlotStyle(dxyNormEtaMeanTrend[i],colors[i]);
    MakeNiceTrendPlotStyle(dxyNormEtaWidthTrend[i],colors[i]);
    MakeNiceTrendPlotStyle(dzNormEtaMeanTrend[i],colors[i]);
    MakeNiceTrendPlotStyle(dzNormEtaWidthTrend[i],colors[i]);
    
    // maps
    
    if(do2DMaps){

      FillMap(dxyMeanMap[i]      ,dxyMapResiduals[i]    ,"mean"); 
      FillMap(dxyWidthMap[i]     ,dxyMapResiduals[i]    ,"width");
      FillMap(dzMeanMap[i]       ,dzMapResiduals[i]     ,"mean"); 
      FillMap(dzWidthMap[i]      ,dzMapResiduals[i]     ,"width");
      
      FillMap(dxyNormMeanMap[i]  ,dxyNormMapResiduals[i],"mean"); 
      FillMap(dxyNormWidthMap[i] ,dxyNormMapResiduals[i],"width");
      FillMap(dzNormMeanMap[i]   ,dzNormMapResiduals[i] ,"mean"); 
      FillMap(dzNormWidthMap[i]  ,dzNormMapResiduals[i] ,"width");
      
      MakeNiceMapStyle(dxyMeanMap[i]);
      MakeNiceMapStyle(dxyWidthMap[i]);     
      MakeNiceMapStyle(dzMeanMap[i]);       
      MakeNiceMapStyle(dzWidthMap[i]);      
      
      MakeNiceMapStyle(dxyNormMeanMap[i]); 
      MakeNiceMapStyle(dxyNormWidthMap[i]); 
      MakeNiceMapStyle(dzNormMeanMap[i]); 
      MakeNiceMapStyle(dzNormWidthMap[i]);  
    }

  }
  

  TString theStrDate       = theDate;
  TString theStrAlignment  = LegLabels[0];

  /*
    // in case labels are needed
    std::vector<TString> vLabels(LegLabels, LegLabels+10);
    vLabels.shrink_to_fit();
  */

  for(Int_t j=1; j < nFiles_; j++) {
    theStrAlignment+=("_vs_"+LegLabels[j]);
  }

  theStrDate.ReplaceAll(" ","");
  theStrAlignment.ReplaceAll(" ","_");

  // DCA absolute
  
  TCanvas *dxyPhiTrend = new TCanvas("dxyPhiTrend","dxyPhiTrend",1200,600);
  arrangeCanvas(dxyPhiTrend,dxyPhiMeanTrend,dxyPhiWidthTrend,nFiles_,LegLabels,theDate,false,setAutoLimits);

  dxyPhiTrend->SaveAs("dxyPhiTrend_"+theStrDate+theStrAlignment+".pdf");
  dxyPhiTrend->SaveAs("dxyPhiTrend_"+theStrDate+theStrAlignment+".png");

  TCanvas *dzPhiTrend = new TCanvas("dzPhiTrend","dzPhiTrend",1200,600);
  arrangeCanvas(dzPhiTrend,dzPhiMeanTrend,dzPhiWidthTrend,nFiles_,LegLabels,theDate,false,setAutoLimits);

  dzPhiTrend->SaveAs("dzPhiTrend_"+theStrDate+theStrAlignment+".pdf");
  dzPhiTrend->SaveAs("dzPhiTrend_"+theStrDate+theStrAlignment+".png");

  TCanvas *dxyEtaTrend = new TCanvas("dxyEtaTrend","dxyEtaTrend",1200,600);
  arrangeCanvas(dxyEtaTrend,dxyEtaMeanTrend,dxyEtaWidthTrend,nFiles_,LegLabels,theDate,false,setAutoLimits);

  dxyEtaTrend->SaveAs("dxyEtaTrend_"+theStrDate+theStrAlignment+".pdf");
  dxyEtaTrend->SaveAs("dxyEtaTrend_"+theStrDate+theStrAlignment+".png");

  TCanvas *dzEtaTrend = new TCanvas("dzEtaTrend","dzEtaTrend",1200,600);
  arrangeCanvas(dzEtaTrend,dzEtaMeanTrend,dzEtaWidthTrend,nFiles_,LegLabels,theDate,false,setAutoLimits);

  dzEtaTrend->SaveAs("dzEtaTrend_"+theStrDate+theStrAlignment+".pdf");
  dzEtaTrend->SaveAs("dzEtaTrend_"+theStrDate+theStrAlignment+".png");

  // fit dz vs phi
  TCanvas *dzPhiTrendFit = new TCanvas("dzPhiTrendFit","dzPhiTrendFit",1200,600);
  arrangeFitCanvas(dzPhiTrendFit,dzPhiMeanTrend,nFiles_,LegLabels,theDate);

  dzPhiTrendFit->SaveAs("dzPhiTrendFit_"+theStrDate+theStrAlignment+".pdf");
  dzPhiTrendFit->SaveAs("dzPhiTrendFit_"+theStrDate+theStrAlignment+".png");

  // DCA normalized

  TCanvas *dxyNormPhiTrend = new TCanvas("dxyNormPhiTrend","dxyNormPhiTrend",1200,600);
  arrangeCanvas(dxyNormPhiTrend,dxyNormPhiMeanTrend,dxyNormPhiWidthTrend,nFiles_,LegLabels,theDate,false,setAutoLimits);

  dxyNormPhiTrend->SaveAs("dxyPhiTrendNorm_"+theStrDate+theStrAlignment+".pdf");
  dxyNormPhiTrend->SaveAs("dxyPhiTrendNorm_"+theStrDate+theStrAlignment+".png");

  TCanvas *dzNormPhiTrend = new TCanvas("dzNormPhiTrend","dzNormPhiTrend",1200,600);
  arrangeCanvas(dzNormPhiTrend,dzNormPhiMeanTrend,dzNormPhiWidthTrend,nFiles_,LegLabels,theDate,false,setAutoLimits);

  dzNormPhiTrend->SaveAs("dzPhiTrendNorm_"+theStrDate+theStrAlignment+".pdf");
  dzNormPhiTrend->SaveAs("dzPhiTrendNorm_"+theStrDate+theStrAlignment+".png");

  TCanvas *dxyNormEtaTrend = new TCanvas("dxyNormEtaTrend","dxyNormEtaTrend",1200,600);
  arrangeCanvas(dxyNormEtaTrend,dxyNormEtaMeanTrend,dxyNormEtaWidthTrend,nFiles_,LegLabels,theDate,false,setAutoLimits);

  dxyNormEtaTrend->SaveAs("dxyEtaTrendNorm_"+theStrDate+theStrAlignment+".pdf");
  dxyNormEtaTrend->SaveAs("dxyEtaTrendNorm_"+theStrDate+theStrAlignment+".png");

  TCanvas *dzNormEtaTrend = new TCanvas("dzNormEtaTrend","dzNormEtaTrend",1200,600);
  arrangeCanvas(dzNormEtaTrend,dzNormEtaMeanTrend,dzNormEtaWidthTrend,nFiles_,LegLabels,theDate,false,setAutoLimits);

  dzNormEtaTrend->SaveAs("dzEtaTrendNorm_"+theStrDate+theStrAlignment+".pdf");
  dzNormEtaTrend->SaveAs("dzEtaTrendNorm_"+theStrDate+theStrAlignment+".png");

  // Bias plots

  TCanvas *BiasesCanvas = new TCanvas("BiasCanvas","BiasCanvas",1200,1200);
  arrangeBiasCanvas(BiasesCanvas,dxyPhiMeanTrend,dzPhiMeanTrend,dxyEtaMeanTrend,dzEtaMeanTrend,nFiles_,LegLabels,theDate,setAutoLimits);
  
  BiasesCanvas->SaveAs("BiasesCanvas_"+theStrDate+theStrAlignment+".pdf");
  BiasesCanvas->SaveAs("BiasesCanvas_"+theStrDate+theStrAlignment+".png");

  TCanvas *dxyPhiBiasCanvas = new TCanvas("dxyPhiBiasCanvas","dxyPhiBiasCanvas",600,600);
  TCanvas *dxyEtaBiasCanvas = new TCanvas("dxyEtaBiasCanvas","dxyEtaBiasCanvas",600,600);
  TCanvas *dzPhiBiasCanvas  = new TCanvas("dzPhiBiasCanvas","dzPhiBiasCanvas",600,600);
  TCanvas *dzEtaBiasCanvas  = new TCanvas("dzEtaBiasCanvas","dzEtaBiasCanvas",600,600);
  
  arrangeCanvas(dxyPhiBiasCanvas,dxyPhiMeanTrend,dxyPhiWidthTrend,nFiles_,LegLabels,theDate,true,setAutoLimits);
  arrangeCanvas(dzPhiBiasCanvas,dzPhiMeanTrend,dzPhiWidthTrend,nFiles_,LegLabels,theDate,true,setAutoLimits);
  arrangeCanvas(dxyEtaBiasCanvas,dxyEtaMeanTrend,dxyEtaWidthTrend,nFiles_,LegLabels,theDate,true,setAutoLimits);
  arrangeCanvas(dzEtaBiasCanvas,dzEtaMeanTrend,dzEtaWidthTrend,nFiles_,LegLabels,theDate,true,setAutoLimits);
  
  dxyPhiBiasCanvas->SaveAs("dxyPhiBiasCanvas_"+theStrDate+theStrAlignment+".pdf");
  dxyEtaBiasCanvas->SaveAs("dxyEtaBiasCanvas_"+theStrDate+theStrAlignment+".pdf");
  dzPhiBiasCanvas->SaveAs("dzPhiBiasCanvas_"+theStrDate+theStrAlignment+".pdf"); 
  dzEtaBiasCanvas->SaveAs("dzEtaBiasCanvas_"+theStrDate+theStrAlignment+".pdf"); 
  
  dxyPhiBiasCanvas->SaveAs("dxyPhiBiasCanvas_"+theStrDate+theStrAlignment+".png");
  dxyEtaBiasCanvas->SaveAs("dxyEtaBiasCanvas_"+theStrDate+theStrAlignment+".png");
  dzPhiBiasCanvas->SaveAs("dzPhiBiasCanvas_"+theStrDate+theStrAlignment+".png"); 
  dzEtaBiasCanvas->SaveAs("dzEtaBiasCanvas_"+theStrDate+theStrAlignment+".png"); 

  
  // 2D Maps

  if(do2DMaps){
  
    TCanvas *dxyAbsMap = new TCanvas("dxyAbsMap","dxyAbsMap",1200,500*nFiles_);
    arrangeCanvas2D(dxyAbsMap,dxyMeanMap,dxyWidthMap,nFiles_,LegLabels,theDate);
    dxyAbsMap->SaveAs("dxyAbsMap_"+theStrDate+theStrAlignment+".pdf");
    dxyAbsMap->SaveAs("dxyAbsMap_"+theStrDate+theStrAlignment+".png");
    
    TCanvas *dzAbsMap = new TCanvas("dzAbsMap","dzAbsMap",1200,500*nFiles_);
    arrangeCanvas2D(dzAbsMap,dzMeanMap,dzWidthMap,nFiles_,LegLabels,theDate);
    dzAbsMap->SaveAs("dzAbsMap_"+theStrDate+theStrAlignment+".pdf");
    dzAbsMap->SaveAs("dzAbsMap_"+theStrDate+theStrAlignment+".png");
    
    TCanvas *dxyNormMap = new TCanvas("dxyNormMap","dxyNormMap",1200,500*nFiles_);
    arrangeCanvas2D(dxyNormMap,dxyNormMeanMap,dxyNormWidthMap,nFiles_,LegLabels,theDate);
    dxyNormMap->SaveAs("dxyNormMap_"+theStrDate+theStrAlignment+".pdf");
    dxyNormMap->SaveAs("dxyNormMap_"+theStrDate+theStrAlignment+".png");

    TCanvas *dzNormMap = new TCanvas("dzNormMap","dzNormMap",1200,500*nFiles_);
    arrangeCanvas2D(dzNormMap,dzNormMeanMap,dzNormWidthMap,nFiles_,LegLabels,theDate);
    dzNormMap->SaveAs("dzNormMap_"+theStrDate+theStrAlignment+".pdf");
    dzNormMap->SaveAs("dzNormMap_"+theStrDate+theStrAlignment+".png");

  }

  timer.Stop(); 	 
  timer.Print();

}

//*************************************************************
void arrangeBiasCanvas(TCanvas *canv,TH1F* dxyPhiMeanTrend[100],TH1F* dzPhiMeanTrend[100],TH1F* dxyEtaMeanTrend[100],TH1F* dzEtaMeanTrend[100],Int_t nFiles, TString LegLabels[10],TString theDate,bool setAutoLimits){
//*************************************************************

  TLegend *lego = new TLegend(0.19,0.82,0.79,0.92);
  // might be useful if many objects are compared
  //lego-> SetNColumns(2);
  lego->SetFillColor(10);
  if(nFiles>4){
    lego->SetTextSize(0.022);
  } else {
    lego->SetTextSize(0.042);
  }
  lego->SetTextFont(42);
  lego->SetFillColor(10);
  lego->SetLineColor(10);
  lego->SetShadowColor(10);

  TPaveText *ptDate =new TPaveText(0.3,0.86,0.79,0.92,"blNDC");
  ptDate->SetFillColor(10);
  ptDate->SetBorderSize(0);
  ptDate->SetLineWidth(0);
  ptDate->SetTextFont(32);
  TText *textDate = ptDate->AddText("Alignment: cosmic rays + 3.8T collisions");
  textDate->SetTextSize(0.04);

  canv->SetFillColor(10);  
  canv->Divide(2,2);
 
  canv->cd(1)->SetBottomMargin(0.14);
  canv->cd(1)->SetLeftMargin(0.18);
  canv->cd(1)->SetRightMargin(0.01);
  canv->cd(1)->SetTopMargin(0.06);  

  canv->cd(2)->SetBottomMargin(0.14);
  canv->cd(2)->SetLeftMargin(0.18);
  canv->cd(2)->SetRightMargin(0.01);
  canv->cd(2)->SetTopMargin(0.06);  
  
  canv->cd(3)->SetBottomMargin(0.14);
  canv->cd(3)->SetLeftMargin(0.18);
  canv->cd(3)->SetRightMargin(0.01);
  canv->cd(3)->SetTopMargin(0.06);  

  canv->cd(4)->SetBottomMargin(0.14);
  canv->cd(4)->SetLeftMargin(0.18);
  canv->cd(4)->SetRightMargin(0.01);
  canv->cd(4)->SetTopMargin(0.06); 

  TH1F *dBiasTrend[4][nFiles]; 
  
  for(Int_t i=0;i<nFiles;i++){
    dBiasTrend[0][i] = dxyPhiMeanTrend[i];
    dBiasTrend[1][i] = dzPhiMeanTrend[i];
    dBiasTrend[2][i] = dxyEtaMeanTrend[i];
    dBiasTrend[3][i] = dzEtaMeanTrend[i];
  }

  Double_t absmin[4]={999.,999.,999.,999.};
  Double_t absmax[4]={-999.,-999.-999.,-999.};

  for(Int_t k=0; k<4; k++){

    canv->cd(k+1);
    
    for(Int_t i=0; i<nFiles; i++){
      if(dBiasTrend[k][i]->GetMaximum()>absmax[k]) absmax[k] = dBiasTrend[k][i]->GetMaximum();
      if(dBiasTrend[k][i]->GetMinimum()<absmin[k]) absmin[k] = dBiasTrend[k][i]->GetMinimum();
    }

    Double_t safeDelta=(absmax[k]-absmin[k])/8.;
    Double_t theExtreme=std::max(absmax[k],TMath::Abs(absmin[k]));

    for(Int_t i=0; i<nFiles; i++){
      if(i==0){

	// if the autoLimits are not set
	if(!setAutoLimits){
	  
	  std::pair<Double_t,Double_t> range = getTheRangeUser(dBiasTrend[k][i],thePlotLimits);
	  dBiasTrend[k][i]->GetYaxis()->SetRangeUser(range.first,range.second);
	  
	} else {
	
	  TString theTitle = dBiasTrend[k][i]->GetName();
	  if( theTitle.Contains("Norm")){
	    dBiasTrend[k][i]->GetYaxis()->SetRangeUser(std::min(-0.48,absmin[k]-safeDelta/2.),std::max(0.48,absmax[k]+safeDelta/2.));
	  } else {
	    dBiasTrend[k][i]->GetYaxis()->SetRangeUser(-theExtreme-(safeDelta/2.),theExtreme+(safeDelta/2.));
	  } 
	}

	dBiasTrend[k][i]->Draw("e1");
	makeNewXAxis(dBiasTrend[k][i]);
	Int_t nbins =  dBiasTrend[k][i]->GetNbinsX();
	Double_t lowedge  = dBiasTrend[k][i]->GetBinLowEdge(1);
	Double_t highedge = dBiasTrend[k][i]->GetBinLowEdge(nbins+1);
	
	TH1F* zeros = DrawZero(dBiasTrend[k][i],nbins,lowedge,highedge,1);
	zeros->Draw("PLsame"); 
	
      }
      else dBiasTrend[k][i]->Draw("e1sames");
      if(k==0){
	lego->AddEntry(dBiasTrend[k][i],LegLabels[i]);
      } 
    }  
  
    lego->Draw();
 
    TPad *current_pad = static_cast<TPad*>(canv->GetPad(k+1));
    CMS_lumi(current_pad,4,33 );

  }
  
}


//*************************************************************
void arrangeCanvas(TCanvas *canv,TH1F* meanplots[100],TH1F* widthplots[100],Int_t nFiles, TString LegLabels[10],TString theDate,bool onlyBias,bool setAutoLimits){
//*************************************************************

  TPaveText *ali = new TPaveText(0.18,0.85,0.50,0.93,"NDC");  
  ali->SetFillColor(10);
  ali->SetTextColor(1);
  ali->SetTextFont(42);
  ali->SetMargin(0.);
  ali->SetLineColor(10);
  ali->SetShadowColor(10);
  TText *alitext = ali->AddText("Alignment: PCL"); 
  alitext->SetTextSize(0.04);

  TLegend *lego = new TLegend(0.18,0.82,0.78,0.92);
  // in case many objects are compared
  // lego-> SetNColumns(2);
  // TLegend *lego = new TLegend(0.18,0.77,0.50,0.86);
  lego->SetFillColor(10);
  if(nFiles>4) {
    lego->SetTextSize(0.02);
  } else {
    lego->SetTextSize(0.04);
  }
  lego->SetTextFont(42);
  lego->SetFillColor(10);
  lego->SetLineColor(10);
  lego->SetShadowColor(10);

  TPaveText *ptDate = NULL;

  canv->SetFillColor(10);  

  if(!onlyBias){ 
    ptDate =new TPaveText(0.3,0.88,0.78,0.91,"blNDC");
  } else {
    ptDate =new TPaveText(0.25,0.88,0.78,0.91,"blNDC");
  }
  
  ptDate->SetFillColor(10);
  ptDate->SetBorderSize(0);
  ptDate->SetLineWidth(0);
  ptDate->SetTextFont(42);
  TText *textDate = ptDate->AddText("Alignment: cosmic rays + 3.8T collisions");
  textDate->SetTextSize(0.04);

  if(!onlyBias) {
    canv->Divide(2,1);
    
    canv->cd(1)->SetBottomMargin(0.14);
    canv->cd(1)->SetLeftMargin(0.17);
    canv->cd(1)->SetRightMargin(0.02);
    canv->cd(1)->SetTopMargin(0.06);  
    
    canv->cd(2)->SetBottomMargin(0.14);
    canv->cd(2)->SetLeftMargin(0.17);
    canv->cd(2)->SetRightMargin(0.02);
    canv->cd(2)->SetTopMargin(0.06);  
    canv->cd(1);

  } else {
    
    canv->cd()->SetBottomMargin(0.14);
    canv->cd()->SetLeftMargin(0.17);
    canv->cd()->SetRightMargin(0.02);
    canv->cd()->SetTopMargin(0.06);  
    canv->cd();
  }

  Double_t absmin(999.);
  Double_t absmax(-999.);

  for(Int_t i=0; i<nFiles; i++){
    if(meanplots[i]->GetMaximum()>absmax) absmax = meanplots[i]->GetMaximum();
    if(meanplots[i]->GetMinimum()<absmin) absmin = meanplots[i]->GetMinimum();
  }

  Double_t safeDelta=(absmax-absmin)/2.;
  Double_t theExtreme=std::max(absmax,TMath::Abs(absmin));

  for(Int_t i=0; i<nFiles; i++){
    
    if(i==0){

      // if the autoLimits are not set
      if(!setAutoLimits){

	std::pair<Double_t,Double_t> range = getTheRangeUser(meanplots[i],thePlotLimits);
	meanplots[i]->GetYaxis()->SetRangeUser(range.first,range.second);

      } else {

	TString theTitle = meanplots[i]->GetName();
	if( theTitle.Contains("Norm")){
	  meanplots[i]->GetYaxis()->SetRangeUser(std::min(-0.48,absmin-safeDelta),std::max(0.48,absmax+safeDelta));
	} else {
	  if(!onlyBias){
	    meanplots[i]->GetYaxis()->SetRangeUser(absmin-safeDelta,absmax+safeDelta);
	  } else {
	    meanplots[i]->GetYaxis()->SetRangeUser(-theExtreme-(TMath::Abs(absmin)/10.),theExtreme+(TMath::Abs(absmax/10.)));
	  }
	}
      }
      
      meanplots[i]->Draw("e1");
      makeNewXAxis(meanplots[i]); 

      if(onlyBias){
	canv->cd();
	Int_t nbins =  meanplots[i]->GetNbinsX();
	Double_t lowedge  = meanplots[i]->GetBinLowEdge(1);
	Double_t highedge = meanplots[i]->GetBinLowEdge(nbins+1);
	
	TH1F* hzero = DrawZero(meanplots[i],nbins,lowedge,highedge,2);
	hzero->Draw("PLsame");
	
      }
    }
    else meanplots[i]->Draw("e1sames");
    lego->AddEntry(meanplots[i],LegLabels[i]); 
  }  
  

  lego->Draw();
  
  //ali->Draw("same");
  //ptDate->Draw("same");

  TPad *current_pad;
  if(!onlyBias){
    current_pad = static_cast<TPad*>(canv->GetPad(1));
  } else {
    current_pad = static_cast<TPad*>(canv->GetPad(0));
  }

  CMS_lumi(current_pad,4,33 );
  // ptDate->Draw("same");

  if(!onlyBias){

    canv->cd(2);
    Double_t absmax2(-999.);
    
    for(Int_t i=0; i<nFiles; i++){
      if(widthplots[i]->GetMaximum()>absmax2) absmax2 = widthplots[i]->GetMaximum();
    }
    
    Double_t safeDelta2=absmax2/3.;
    
    for(Int_t i=0; i<nFiles; i++){
      
      widthplots[i]->GetXaxis()->SetLabelOffset(999);
      widthplots[i]->GetXaxis()->SetTickLength(0);
      
      if(i==0){ 

	if(!setAutoLimits){
	  std::pair<Double_t,Double_t> range = getTheRangeUser(widthplots[i],thePlotLimits);
	  widthplots[i]->GetYaxis()->SetRangeUser(range.first,range.second);
	} else {
	  widthplots[i]->SetMinimum(0.5);
	  widthplots[i]->SetMaximum(absmax2+safeDelta2);
	}

	widthplots[i]->Draw("e1");
	makeNewXAxis(widthplots[i]);
      } else widthplots[i]->Draw("e1sames");
    }
    
    lego->Draw();
   
    TPad *current_pad2 = static_cast<TPad*>(canv->GetPad(2));
    CMS_lumi(current_pad2,4,33 );
    //ptDate->Draw("same");

  }
}

//*************************************************************
void arrangeCanvas2D(TCanvas *canv,TH2F* meanmaps[100],TH2F* widthmaps[100],Int_t nFiles,TString LegLabels[10],TString theDate)
//*************************************************************
{
  TLegend *lego = new TLegend(0.18,0.75,0.58,0.92);
  lego->SetFillColor(10);
  lego->SetTextSize(0.05);
  lego->SetTextFont(42);
  lego->SetFillColor(10);
  lego->SetLineColor(10);
  lego->SetShadowColor(10);

  TPaveText *pt[nFiles];
  TPaveText *pt2[nFiles];
  TPaveText *pt3[nFiles];

  for(Int_t i=0; i<nFiles; i++){
    pt[i] = new TPaveText(0.13,0.95,0.191,0.975,"NDC");
    //pt[i] = new TPaveText(gPad->GetUxmin(),gPad->GetUymax()+0.3,gPad->GetUxmin()+0.6,gPad->GetUymax()+0.3,"NDC");
    //std::cout<<"gPad->GetUymax():"<<gPad->GetUymax()<<std::endl;
    //pt[i] = new TPaveText(gPad->GetLeftMargin(),0.95,gPad->GetLeftMargin()+0.3,0.98,"NDC");
    pt[i]->SetFillColor(10);
    pt[i]->SetTextColor(1);
    pt[i]->SetTextFont(61);
    pt[i]->SetTextAlign(22);
    TText *text1 = pt[i]->AddText("CMS"); // preliminary 2015 p-p data, #sqrt{s}=8 TeV "+LegLabels[i]);
    text1->SetTextSize(0.05);
    //delete text1;

    float extraOverCmsTextSize  = 0.76;

    pt2[i] = new TPaveText(0.21,0.95,0.25,0.975,"NDC");
    pt2[i]->SetFillColor(10);
    pt2[i]->SetTextColor(1);
    //pt[i]->SetTextSize(0.05);
    pt2[i]->SetTextFont(52);
    pt2[i]->SetTextAlign(12);
    // TText *text2 = pt2->AddText("run: "+theDate);
    TText *text2 = pt2[i]->AddText(toTString(PRELIMINARY));
    text2->SetTextSize(0.06*extraOverCmsTextSize); 
    
    pt3[i] = new TPaveText(0.55,0.955,0.95,0.98,"NDC");
    pt3[i]->SetFillColor(10);
    pt3[i]->SetTextColor(kBlue);
    pt3[i]->SetTextFont(61);
    pt3[i]->SetTextAlign(22);
    // TText *text2 = pt2->AddText("run: "+theDate);
    TText *text3 = pt3[i]->AddText(LegLabels[i]);
    text3->SetTextSize(0.05); 

  }

  canv->SetFillColor(10);  
  canv->Divide(2,nFiles);
   
  Double_t absmin(999.);
  Double_t absmax(-999.);
  
  for(Int_t i=0; i<nFiles; i++){
    if(meanmaps[i]->GetMaximum()>absmax) absmax = meanmaps[i]->GetMaximum();
    if(meanmaps[i]->GetMinimum()<absmin) absmin = meanmaps[i]->GetMinimum();
  }

  for(Int_t i=0; i<nFiles; i++){
 
    canv->cd(2*i+1)->SetBottomMargin(0.13);
    canv->cd(2*i+1)->SetLeftMargin(0.12);
    canv->cd(2*i+1)->SetRightMargin(0.19);
    canv->cd(2*i+1)->SetTopMargin(0.08);  

    meanmaps[i]->GetZaxis()->SetRangeUser(absmin,absmax);
    meanmaps[i]->Draw("colz");
    makeNewPairOfAxes(meanmaps[i]);

    pt[i]->Draw("same");
    pt2[i]->Draw("same");
    pt3[i]->Draw("same");

    canv->cd(2*(i+1))->SetBottomMargin(0.13);
    canv->cd(2*(i+1))->SetLeftMargin(0.12);
    canv->cd(2*(i+1))->SetRightMargin(0.19);
    canv->cd(2*(i+1))->SetTopMargin(0.08);  

    widthmaps[i]->Draw("colz");
    makeNewPairOfAxes(widthmaps[i]);

    pt[i]->Draw("same");
    pt2[i]->Draw("same");
    pt3[i]->Draw("same");

  }
}

//*************************************************************
void arrangeFitCanvas(TCanvas *canv,TH1F* meanplots[100],Int_t nFiles, TString LegLabels[10],TString theDate)
//*************************************************************
{
  canv->SetBottomMargin(0.14);
  canv->SetLeftMargin(0.1);
  canv->SetRightMargin(0.02);
  canv->SetTopMargin(0.08);  

  TLegend *lego = new TLegend(0.12,0.75,0.82,0.90);
  lego->SetFillColor(10);
  lego->SetTextSize(0.045);
  lego->SetTextFont(42);
  lego->SetFillColor(10);
  lego->SetLineColor(10);
  //lego->SetNColumns(2);
  lego->SetShadowColor(10);

  TF1 *fleft[nFiles]; 
  TF1 *fright[nFiles];
  TF1 *fall[nFiles];  

  TF1 *FitDzUp[nFiles];
  TF1 *FitDzDown[nFiles];

  for(Int_t j=0;j<nFiles;j++){
    
    Double_t deltaZ(0);
    Double_t sigmadeltaZ(-1);

    TCanvas *theNewCanvas2 = new TCanvas("NewCanvas2","Fitting Canvas 2",800,600);
    theNewCanvas2->Divide(2,1);

    TH1F *hnewUp   = (TH1F*)meanplots[j]->Clone("hnewUp_dz_phi");
    TH1F *hnewDown = (TH1F*)meanplots[j]->Clone("hnewDown_dz_phi");
    
    fleft[j]  = new TF1(Form("fleft_%i",j),fULine,_boundMin,_boundSx,1);
    fright[j] = new TF1(Form("fright_%i",j),fULine,_boundDx,_boundMax,1);
    fall[j]   = new TF1(Form("fall_%i",j),fDLine,_boundSx,_boundDx,1);
    
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
      MakeNiceTF1Style(fright[j],my_colors[j]);
      MakeNiceTF1Style(fleft[j],my_colors[j]);
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
      MakeNiceTF1Style(fall[j],my_colors[j]);
      fall[j]->Draw("same");
      canv->cd();
      hnewUp->GetYaxis()->SetTitleOffset(0.7);
      if(j==0){
	hnewUp->Draw();
	makeNewXAxis(hnewUp);
      } else {
	hnewUp->Draw("same");
	makeNewXAxis(hnewUp);
      } 
      fright[j]->Draw("sames");
      fleft[j]->Draw("same");
      fall[j]->Draw("same");
    }
    
    if(j==nFiles-1){
      theNewCanvas2->Close();
    }
    
    deltaZ=(fright[j]->GetParameter(0) - fall[j]->GetParameter(0))/2;
    sigmadeltaZ=0.5*TMath::Sqrt(fright[j]->GetParError(0)*fright[j]->GetParError(0) + fall[j]->GetParError(0)*fall[j]->GetParError(0));
    TString COUT = Form(" : #Delta z = %.f #pm %.f #mum",deltaZ,sigmadeltaZ);
    
    lego->AddEntry(meanplots[j],LegLabels[j]+COUT); 

    if(j==nFiles-1){ 
      outfile <<deltaZ<<"|"<<sigmadeltaZ<<endl;
    }
    
    delete theNewCanvas2;

  }
 
  //TkAlStyle::drawStandardTitle(Coll0T15);
  lego->Draw("same");
  CMS_lumi( canv,4,33 );
  //ptDate->Draw("same");
  //pt->Draw("same");

}

//*************************************************************
std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > fitStudentTResiduals(TH1 *hist)
//*************************************************************
{

  hist->SetMarkerStyle(21);
  hist->SetMarkerSize(0.8);
  hist->SetStats(1);
 
  double dx = hist->GetBinWidth(1);
  double nmax = hist->GetBinContent(hist->GetMaximumBin());
  double xmax = hist->GetBinCenter(hist->GetMaximumBin());
  double nn = 7*nmax;
  
  int nb = hist->GetNbinsX();
  double n1 = hist->GetBinContent(1);
  double n9 = hist->GetBinContent(nb);
  double bg = 0.5*(n1+n9);
  
  double x1 = hist->GetBinCenter(1);
  double x9 = hist->GetBinCenter(nb);
  
  // create a TF1 with the range from x1 to x9 and 5 parameters
  
  TF1 *tp0Fcn = new TF1("tmp", tp0Fit, x1, x9, 5 );
  
  tp0Fcn->SetParName( 0, "mean" );
  tp0Fcn->SetParName( 1, "sigma" );
  tp0Fcn->SetParName( 2, "nu" );
  tp0Fcn->SetParName( 3, "area" );
  tp0Fcn->SetParName( 4, "BG" );
  
  tp0Fcn->SetNpx(500);
  tp0Fcn->SetLineWidth(2);
  //tp0Fcn->SetLineColor(kMagenta);
  //tp0Fcn->SetLineColor(kGreen);
  tp0Fcn->SetLineColor(kRed);

  // set start values for some parameters:
    
  tp0Fcn->SetParameter( 0, xmax ); // peak position
  tp0Fcn->SetParameter( 1, 4*dx ); // width
  tp0Fcn->SetParameter( 2, 2.2 ); // nu
  tp0Fcn->SetParameter( 3, nn ); // N
  tp0Fcn->SetParameter( 4, bg );
  
  hist->Fit( "tmp", "R", "ep" );
  // h->Fit("tmp","V+","ep");
  
  hist->Draw("histepsame");  // data again on top
  
  float res_mean  = tp0Fcn->GetParameter(0);
  float res_width = tp0Fcn->GetParameter(1);
  
  float res_mean_err  = tp0Fcn->GetParError(0);
  float res_width_err = tp0Fcn->GetParError(1);

  std::pair<Double_t,Double_t> resultM;
  std::pair<Double_t,Double_t> resultW;

  resultM = std::make_pair(res_mean,res_mean_err);
  resultW = std::make_pair(res_width,res_width_err);

  std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > result;
  
  result = std::make_pair(resultM,resultW);
  return result;

}

//*************************************************************
Double_t tp0Fit( Double_t *x, Double_t *par5 ) 
//*************************************************************
{
  static int nn = 0;
  nn++;
  static double dx = 0.1;
  static double b1 = 0;
  if( nn == 1 ) b1 = x[0];
  if( nn == 2 ) dx = x[0] - b1;
  //
  //--  Mean and width:
  //
  double xm = par5[0];
  double t = ( x[0] - xm ) / par5[1];
  double tt = t*t;
  //
  //--  exponent:
  //
  double rn = par5[2];
  double xn = 0.5 * ( rn + 1.0 );
  //
  //--  Normalization needs Gamma function:
  //
  double pk = 0.0;

  if( rn > 0.0 ) {

    double pi = 3.14159265358979323846;
    double aa = dx / par5[1] / sqrt(rn*pi) * TMath::Gamma(xn) / TMath::Gamma(0.5*rn);

    pk = par5[3] * aa * exp( -xn * log( 1.0 + tt/rn ) );
  }

  return pk + par5[4];
}

//*************************************************************
std::pair<Double_t,Double_t> getMedian(TH1F *histo)
//*************************************************************
{
  Double_t median = 999;
  int nbins = histo->GetNbinsX();

  //extract median from histogram
  double *x = new double[nbins];
  double *y = new double[nbins];
  for (int j = 0; j < nbins; j++) {
    x[j] = histo->GetBinCenter(j+1);
    y[j] = histo->GetBinContent(j+1);
  }
  median = TMath::Median(nbins, x, y);
  
  delete[] x; x = 0;
  delete [] y; y = 0;  

  std::pair<Double_t,Double_t> result;
  result = std::make_pair(median,median/TMath::Sqrt(histo->GetEntries()));

  return result;

}

//*************************************************************
std::pair<Double_t,Double_t> getMAD(TH1F *histo)
//*************************************************************
{

  int nbins = histo->GetNbinsX();
  Double_t median = getMedian(histo).first;
  Double_t x_lastBin = histo->GetBinLowEdge(nbins+1);
  const char *HistoName =histo->GetName();
  TString Finalname = Form("resMed%s",HistoName);
  TH1F *newHisto = new TH1F(Finalname,Finalname,nbins,0.,x_lastBin);
  Double_t *residuals = new Double_t[nbins];
  Double_t *weights = new Double_t[nbins];

  for (int j = 0; j < nbins; j++) {
    residuals[j] = TMath::Abs(median - histo->GetBinCenter(j+1));
    weights[j]=histo->GetBinContent(j+1);
    newHisto->Fill(residuals[j],weights[j]);
  }
  
  Double_t theMAD = (getMedian(newHisto).first)*1.4826;
  newHisto->Delete("");
  
  std::pair<Double_t,Double_t> result;
  result = std::make_pair(theMAD,theMAD/histo->GetEntries());

  return result;

}

//*************************************************************
std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > fitResiduals(TH1 *hist)
//*************************************************************
{
  //float fitResult(9999);
  //if (hist->GetEntries() < 20) return ;
  
  float mean  = hist->GetMean();
  float sigma = hist->GetRMS();
  
  if(TMath::IsNaN(mean) || TMath::IsNaN(sigma)){  
    mean=0;
    sigma= - hist->GetXaxis()->GetBinLowEdge(1) + hist->GetXaxis()->GetBinLowEdge(hist->GetNbinsX()+1) ;
  }

  TF1 func("tmp", "gaus", mean - 2.*sigma, mean + 2.*sigma); 
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

  float res_mean  = func.GetParameter(1);
  float res_width = func.GetParameter(2);
  
  float res_mean_err  = func.GetParError(1);
  float res_width_err = func.GetParError(2);

  std::pair<Double_t,Double_t> resultM;
  std::pair<Double_t,Double_t> resultW;

  resultM = std::make_pair(res_mean,res_mean_err);
  resultW = std::make_pair(res_width,res_width_err);

  std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > result;
  
  result = std::make_pair(resultM,resultW);
  return result;
}

//*************************************************************
Double_t DoubleSidedCB(double* x, double* par){
//*************************************************************

  double m      = x[0];
  double m0     = par[0]; 
  double sigma  = par[1];
  double alphaL = par[2];
  double alphaR = par[3];
  double nL     = par[4];
  double nR     = par[5];
  double N      = par[6];

  Double_t arg = m - m0;
  
  if (arg < 0.0) {
    Double_t t = (m-m0)/sigma; //t < 0
    Double_t absAlpha = fabs((Double_t)alphaL); //slightly redundant since alpha > 0 anyway, but never mind
    if (t >= -absAlpha) { //-absAlpha <= t < 0
      return N*exp(-0.5*t*t);
    } else {
      Double_t a = TMath::Power(nL/absAlpha,nL)*exp(-0.5*absAlpha*absAlpha);
      Double_t b = nL/absAlpha - absAlpha;
      return N*(a/TMath::Power(b - t, nL)); //b - t
    }
  } else {
    Double_t t = (m-m0)/sigma; //t > 0
    Double_t absAlpha = fabs((Double_t)alphaR);
    if (t <= absAlpha) { //0 <= t <= absAlpha
      return N*exp(-0.5*t*t);
    } else {
      Double_t a = TMath::Power(nR/absAlpha,nR)*exp(-0.5*absAlpha*absAlpha);
      Double_t b = nR/absAlpha - absAlpha;   
      return N*(a/TMath::Power(b + t, nR)); //b + t
    }
  }
}

//*************************************************************
std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > fitResidualsCB(TH1 *hist)
//*************************************************************
{
  
  //hist->Rebin(2);

  float mean  = hist->GetMean();
  float sigma = hist->GetRMS();
  //int   nbinsX   = hist->GetNbinsX();
  float nentries = hist->GetEntries();
  float meanerr  = sigma/TMath::Sqrt(nentries);
  float sigmaerr = TMath::Sqrt(sigma*sigma*TMath::Sqrt(2/nentries));

  float lowBound  = hist->GetXaxis()->GetBinLowEdge(1);
  float highBound = hist->GetXaxis()->GetBinLowEdge(hist->GetNbinsX()+1);

  if(TMath::IsNaN(mean) || TMath::IsNaN(sigma)){  
    mean=0;
    sigma= - lowBound + highBound;
  }
  
  TF1 func("tmp", "gaus", mean - 1.*sigma, mean + 1.*sigma); 
  if (0 == hist->Fit(&func,"QNR")) { // N: do not blow up file by storing fit!
    mean  = func.GetParameter(1);
    sigma = func.GetParameter(2);
  }
  
  // first round
  TF1 *doubleCB = new TF1("myDoubleCB",DoubleSidedCB,lowBound,highBound,7);
  doubleCB->SetParameters(mean,sigma,1.5,1.5,2.5,2.5,100);
  doubleCB->SetParLimits(0,mean-meanerr,mean+meanerr);
  doubleCB->SetParLimits(1,0.,sigma+2*sigmaerr);
  doubleCB->SetParLimits(2,0.,30.);
  doubleCB->SetParLimits(3,0.,30.);
  doubleCB->SetParLimits(4,0.,50.);
  doubleCB->SetParLimits(5,0.,50.);
  doubleCB->SetParLimits(6,0.,100*nentries);

  doubleCB->SetParNames("#mu","#sigma","#alpha_{L}","#alpha_{R}","n_{L}","n_{R}","N");
  doubleCB->SetLineColor(kRed);
  doubleCB->SetNpx(1000);
  // doubleCB->SetRange(0.8*lowBound,0.8*highBound);

  hist->Fit(doubleCB,"QM");

  // second round

  float p0 = doubleCB->GetParameter(0);
  float p1 = doubleCB->GetParameter(1);
  float p2 = doubleCB->GetParameter(2);
  float p3 = doubleCB->GetParameter(3);
  float p4 = doubleCB->GetParameter(4);
  float p5 = doubleCB->GetParameter(5);
  float p6 = doubleCB->GetParameter(6);
  
  float p0err = doubleCB->GetParError(0);
  float p1err = doubleCB->GetParError(1);
  float p2err = doubleCB->GetParError(2);
  float p3err = doubleCB->GetParError(3);
  float p4err = doubleCB->GetParError(4);
  float p5err = doubleCB->GetParError(5);
  float p6err = doubleCB->GetParError(6);

  if( (doubleCB->GetChisquare()/doubleCB->GetNDF()) >5){

    std::cout<<"------------------------"<<std::endl;
    std::cout<<"chi2 1st:"<<doubleCB->GetChisquare()<<std::endl;

    //std::cout<<"p0: "<<p0<<"+/-"<<p0err<<std::endl;
    //std::cout<<"p1: "<<p1<<"+/-"<<p1err<<std::endl;
    //std::cout<<"p2: "<<p2<<"+/-"<<p2err<<std::endl;
    //std::cout<<"p3: "<<p3<<"+/-"<<p3err<<std::endl;
    //std::cout<<"p4: "<<p4<<"+/-"<<p4err<<std::endl;
    //std::cout<<"p5: "<<p5<<"+/-"<<p5err<<std::endl;
    //std::cout<<"p6: "<<p6<<"+/-"<<p6err<<std::endl;

    doubleCB->SetParameters(p0,p1,3,3,6,6,p6);
    doubleCB->SetParLimits(0,p0-2*p0err,p0+2*p0err);
    doubleCB->SetParLimits(1,p1-2*p1err,p0+2*p1err);
    doubleCB->SetParLimits(2,p2-2*p2err,p0+2*p2err);
    doubleCB->SetParLimits(3,p3-2*p3err,p0+2*p3err);
    doubleCB->SetParLimits(4,p4-2*p4err,p0+2*p4err);
    doubleCB->SetParLimits(5,p5-2*p5err,p0+2*p5err);
    doubleCB->SetParLimits(6,p6-2*p6err,p0+2*p6err);

    hist->Fit(doubleCB,"MQ");

    //gMinuit->Command("SCAn 1");
    //TGraph *gr = (TGraph*)gMinuit->GetPlot();
    //gr->SetMarkerStyle(21);
    //gr->Draw("alp"); 

    std::cout<<"chi2 2nd:"<<doubleCB->GetChisquare()<<std::endl;
    
  }

  float res_mean  = doubleCB->GetParameter(0);
  float res_width = doubleCB->GetParameter(1);
  
  float res_mean_err  = doubleCB->GetParError(0);
  float res_width_err = doubleCB->GetParError(1);

  std::pair<Double_t,Double_t> resultM;
  std::pair<Double_t,Double_t> resultW;

  resultM = std::make_pair(res_mean,res_mean_err);
  resultW = std::make_pair(res_width,res_width_err);

  std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > result;
  
  result = std::make_pair(resultM,resultW);
  return result;

}

//*************************************************************
void FillTrendPlot(TH1F* trendPlot, TH1F* residualsPlot[100], TString fitPar_, TString var_,Int_t myBins_)
//*************************************************************
{

  //std::cout<<"trendPlot name: "<<trendPlot->GetName()<<std::endl;

  // float phiInterval = (360.)/myBins_;
  float phiInterval = (2*TMath::Pi()/myBins_);
  float etaInterval = 5./myBins_;
 
  for ( int i=0; i<myBins_; ++i ) {
    
    //int binn = i+1;

    char phipositionString[129];
    // float phiposition = (-180+i*phiInterval)+(phiInterval/2);
    float phiposition = (-TMath::Pi()+i*phiInterval)+(phiInterval/2);
    sprintf(phipositionString,"%.1f",phiposition);
    
    char etapositionString[129];
    float etaposition = (-2.5+i*etaInterval)+(etaInterval/2);
    sprintf(etapositionString,"%.1f",etaposition);

    std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t> > myFit = std::make_pair(std::make_pair(0.,0.),std::make_pair(0.,0.));

    if ( ((TString)trendPlot->GetName()).Contains("Norm") ) {
      //std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > myFit = fitResiduals(residualsPlot[i]);
      myFit = fitResiduals(residualsPlot[i]);
    } else {
      //std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > myFit = fitStudentTResiduals(residualsPlot[i]);
      myFit = fitResiduals(residualsPlot[i]);
      //myFit = fitStudentTResiduals(residualsPlot[i]);
    }

    if(fitPar_=="mean"){
      float mean_      = myFit.first.first;
      float meanErr_   = myFit.first.second;
      trendPlot->SetBinContent(i+1,mean_);
      trendPlot->SetBinError(i+1,meanErr_);
    } else if (fitPar_=="width"){
      float width_     = myFit.second.first;
      float widthErr_  = myFit.second.second;
      trendPlot->SetBinContent(i+1,width_);
      trendPlot->SetBinError(i+1,widthErr_);
    } else if (fitPar_=="median"){
      float median_    = getMedian(residualsPlot[i]).first;
      float medianErr_ = getMedian(residualsPlot[i]).second;
      trendPlot->SetBinContent(i+1,median_);
      trendPlot->SetBinError(i+1,medianErr_);
    } else if (fitPar_=="mad"){
      float mad_       = getMAD(residualsPlot[i]).first; 
      float madErr_    = getMAD(residualsPlot[i]).second;
      trendPlot->SetBinContent(i+1,mad_);
      trendPlot->SetBinError(i+1,madErr_);
    } else {
      std::cout<<"PrimaryVertexValidation::FillTrendPlot() "<<fitPar_<<" unknown estimator!"<<std::endl;
    }
  }

  //trendPlot->GetXaxis()->LabelsOption("h");

  if(fitPar_=="mean" || fitPar_=="median"){

    TString res;
    if(TString(residualsPlot[0]->GetName()).Contains("dxy")) res="dxy";
    else if(TString(residualsPlot[0]->GetName()).Contains("dz")) res="dz";
    else if(TString(residualsPlot[0]->GetName()).Contains("IP2D")) res="IP2D";
    else if(TString(residualsPlot[0]->GetName()).Contains("resz")) res="resz";
    
    TCanvas *fitOutput = new TCanvas(Form("fitOutput_%s_%s_%s",res.Data(),var_.Data(),trendPlot->GetName()),Form("fitOutput_%s_%s",res.Data(),var_.Data()),1200,1200);
    fitOutput->Divide(5,5);
    
    TCanvas *fitPulls = new TCanvas(Form("fitPulls_%s_%s_%s",res.Data(),var_.Data(),trendPlot->GetName()),Form("fitPulls_%s_%s",res.Data(),var_.Data()),1200,1200);
    fitPulls->Divide(5,5);

    TH1F* residualsPull[myBins_];

    for(Int_t i=0;i<myBins_;i++){
      
      TF1 *tmp1 = (TF1*)residualsPlot[i]->GetListOfFunctions()->FindObject("tmp");
      fitOutput->cd(i+1)->SetLogy();
      fitOutput->cd(i+1)->SetBottomMargin(0.16);
      //fitOutput->cd(i+1)->SetTopMargin(0.05);
      //residualsPlot[i]->Sumw2();
      MakeNicePlotStyle(residualsPlot[i]);
      residualsPlot[i]->SetMarkerStyle(20);
      residualsPlot[i]->SetMarkerSize(1.);
      residualsPlot[i]->SetStats(0);
      //residualsPlot[i]->GetXaxis()->SetRangeUser(-3*(tmp1->GetParameter(1)),3*(tmp1->GetParameter(1)));
      residualsPlot[i]->Draw("e1");
      residualsPlot[i]->GetYaxis()->UnZoom();

      //std::cout<<"*********************"<<std::endl;
      //std::cout<<"fitOutput->cd("<<i+1<<")"<<std::endl;
      //std::cout<<"residualsPlot["<<i<<"]->GetTitle() = "<<residualsPlot[i]->GetTitle()<<std::endl;
      
      // -- for chi2 ----
      TPaveText *pt = new TPaveText(0.13,0.78,0.33,0.88,"NDC");
      pt->SetFillColor(10);
      pt->SetTextColor(1);
      pt->SetTextSize(0.07);
      pt->SetTextFont(42);
      pt->SetTextAlign(22);

      //TF1 *tmp1 = (TF1*)residualsPlot[i]->GetListOfFunctions()->FindObject("tmp");
      TString COUT = Form("#chi^{2}/ndf=%.1f",tmp1->GetChisquare()/tmp1->GetNDF());
      
      TText *text1 = pt->AddText(COUT);
      text1->SetTextFont(72);
      text1->SetTextColor(kBlue);
      pt->Draw("same");
    
      // -- for bins --
     
      TPaveText *title = new TPaveText(0.1,0.93,0.8,0.95,"NDC");
      title->SetFillColor(10);
      title->SetTextColor(1);
      title->SetTextSize(0.07);
      title->SetTextFont(42);
      title->SetTextAlign(22);
 
      //TText *text2 = title->AddText(residualsPlot[i]->GetTitle());
      //text2->SetTextFont(72);
      //text2->SetTextColor(kBlue);

      title->Draw("same");

      fitPulls->cd(i+1);
      fitPulls->cd(i+1)->SetBottomMargin(0.15);
      fitPulls->cd(i+1)->SetLeftMargin(0.15); 
      fitPulls->cd(i+1)->SetRightMargin(0.05); 
 
      residualsPull[i]=(TH1F*)residualsPlot[i]->Clone(Form("pull_%s",residualsPlot[i]->GetName()));
      for(Int_t nbin=1;nbin<=residualsPull[i]->GetNbinsX(); nbin++){
      	if(residualsPlot[i]->GetBinContent(nbin)!=0){ 
	  residualsPull[i]->SetBinContent(nbin,(residualsPlot[i]->GetBinContent(nbin) - tmp1->Eval(residualsPlot[i]->GetBinCenter(nbin)))/residualsPlot[i]->GetBinContent(nbin));
	  residualsPull[i]->SetBinError(nbin,0.1);
	}
      }

      TF1* toDel = (TF1*)residualsPull[i]->FindObject("tmp");
      if(toDel) residualsPull[i]->GetListOfFunctions()->Remove(toDel); 
      residualsPull[i]->SetMarkerStyle(20);
      residualsPull[i]->SetMarkerSize(1.);
      residualsPull[i]->SetStats(0);
			
      residualsPull[i]->GetYaxis()->SetTitle("(res-fit)/res");
      // residualsPull[i]->SetOptTitle(1);
      residualsPull[i]->GetXaxis()->SetLabelFont(42);
      residualsPull[i]->GetYaxis()->SetLabelFont(42);
      residualsPull[i]->GetYaxis()->SetLabelSize(.07);
      residualsPull[i]->GetXaxis()->SetLabelSize(.07);
      residualsPull[i]->GetYaxis()->SetTitleSize(.07);
      residualsPull[i]->GetXaxis()->SetTitleSize(.07);
      residualsPull[i]->GetXaxis()->SetTitleOffset(0.9);
      residualsPull[i]->GetYaxis()->SetTitleOffset(1.2);
      residualsPull[i]->GetXaxis()->SetTitleFont(42);
      residualsPull[i]->GetYaxis()->SetTitleFont(42);
      
      residualsPull[i]->Draw("e1");
      residualsPull[i]->GetYaxis()->UnZoom();
    }
    
    
    TString tpName =trendPlot->GetName();

    TString FitNameToSame  = Form("fitOutput_%s",(tpName.ReplaceAll("means_","").Data()));
    //fitOutput->SaveAs(FitNameToSame+".pdf");
    //fitOutput->SaveAs(FitNameToSame+".png");
    TString PullNameToSave = Form("fitPulls_%s",(tpName.ReplaceAll("means_","").Data()));
    //fitPulls->SaveAs(PullNameToSave+".pdf");
    //fitPulls->SaveAs(PullNameToSave+".png");
    
    //fitOutput->SaveAs(Form("fitOutput_%s_%s_%s.pdf",res.Data(),var_.Data(),trendPlot->GetName()));
    fitOutput->SaveAs(Form("fitOutput_%s.pdf",(((TString)trendPlot->GetName()).ReplaceAll("means_","")).Data()));
    fitPulls->SaveAs(Form("fitPulls_%s.pdf",(((TString)trendPlot->GetName()).ReplaceAll("means_","")).Data()));
    //fitOutput->SaveAs(Form("fitOutput_%s.png",(((TString)trendPlot->GetName()).ReplaceAll("means_","")).Data()));

  }
}

//*************************************************************
void FillMap(TH2F* trendMap, TH1F* residualsMapPlot[48][48], TString fitPar_)
//*************************************************************
{
 
  float phiInterval = (360.)/nBins_;
  float etaInterval = 5./nBins_;

  for ( int i=0; i<nBins_; ++i ) {
    
    char phipositionString[129];
    float phiposition = (-180+i*phiInterval)+(phiInterval/2);
    sprintf(phipositionString,"%.f",phiposition);
    
    trendMap->GetYaxis()->SetBinLabel(i+1,phipositionString); 

    for ( int j=0; j<nBins_; ++j ) {
      
      //cout<<"(i,j)="<<i<<","<<j<<endl;

      char etapositionString[129];
      float etaposition = (-2.5+j*etaInterval)+(etaInterval/2);
      sprintf(etapositionString,"%.1f",etaposition);

      if(i==0) { trendMap->GetXaxis()->SetBinLabel(j+1,etapositionString); }

      if(fitPar_=="mean"){
	float mean_      = fitResiduals(residualsMapPlot[i][j]).first.first;
	float meanErr_   = fitResiduals(residualsMapPlot[i][j]).first.second;
	//std::cout<<"bin i: "<<i<<" bin j: "<<j<<" mean: "<<mean_<<"+/-"<<meanErr_<<endl;
	trendMap->SetBinContent(j+1,i+1,mean_);
	trendMap->SetBinError(j+1,i+1,meanErr_);
      } else if (fitPar_=="width"){
	float width_     = fitResiduals(residualsMapPlot[i][j]).second.first;
	float widthErr_  = fitResiduals(residualsMapPlot[i][j]).second.second;
	trendMap->SetBinContent(j+1,i+1,width_);
	trendMap->SetBinError(j+1,i+1,widthErr_);
	//std::cout<<"bin i: "<<i<<" bin j: "<<j<<" width: "<<width_<<"+/-"<<widthErr_<<endl;
      } else if (fitPar_=="median"){
	float median_    = getMedian(residualsMapPlot[i][j]).first;
	float medianErr_ = getMedian(residualsMapPlot[i][j]).second;
	trendMap->SetBinContent(j+1,i+1,median_);
	trendMap->SetBinError(j+1,i+1,medianErr_);
      } else if (fitPar_=="mad"){
	float mad_       = getMAD(residualsMapPlot[i][j]).first; 
	float madErr_    = getMAD(residualsMapPlot[i][j]).second;
	trendMap->SetBinContent(j+1,i+1,mad_);
	trendMap->SetBinError(j+1,i+1,madErr_);
      } else {
	std::cout<<"FitPVResiduals::FillMap() "<<fitPar_<<" unknown estimator!"<<std::endl;
      }   
    } // closes loop on eta bins
  } // cloeses loop on phi bins
}

/*--------------------------------------------------------------------*/
void  MakeNiceTrendPlotStyle(TH1 *hist,Int_t color)
/*--------------------------------------------------------------------*/
{ 

  Int_t markers[9] = {kFullSquare,kFullCircle,kDot,kFullTriangleDown,kOpenSquare,kOpenCircle,kFullTriangleDown,kFullTriangleUp,kOpenTriangleDown};
  
  // color for approval
  //Int_t colors[8]  = {kBlack,kGreen+2,kRed,kGreen+2,kOrange,kMagenta,kCyan,kViolet};

  Int_t colors[10]={kBlack,kRed,kBlue,kMagenta,kBlack,kRed,kBlue,kGreen};
 
  //Int_t markers[4] = {kFullSquare,kFullCircle,kOpenSquare};
  //Int_t colors[4]  = {kBlack,kRed,kBlue};

  hist->SetStats(kFALSE);  
  hist->SetLineWidth(2);
  hist->GetXaxis()->CenterTitle(true);
  hist->GetYaxis()->CenterTitle(true);
  hist->GetXaxis()->SetTitleFont(42); 
  hist->GetYaxis()->SetTitleFont(42);  
  hist->GetXaxis()->SetTitleSize(0.065);
  hist->GetYaxis()->SetTitleSize(0.065);
  hist->GetXaxis()->SetTitleOffset(1.0);
  hist->GetYaxis()->SetTitleOffset(1.2);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelSize(.05);
  hist->GetXaxis()->SetLabelSize(.07);
  //hist->GetXaxis()->SetNdivisions(505);
  if(color!=8){
    hist->SetMarkerSize(1.5);
  } else {
    hist->SetLineWidth(3);
    hist->SetMarkerSize(0.0);    
  }
  hist->SetMarkerStyle(markers[color]);
  hist->SetLineColor(colors[color]);
  hist->SetMarkerColor(colors[color]);

}

/*--------------------------------------------------------------------*/
void MakeNicePlotStyle(TH1 *hist)
/*--------------------------------------------------------------------*/
{ 
  hist->SetStats(kFALSE);  
  hist->SetLineWidth(2);
  hist->GetXaxis()->SetNdivisions(505);
  hist->GetXaxis()->CenterTitle(true);
  hist->GetYaxis()->CenterTitle(true);
  hist->GetXaxis()->SetTitleFont(42); 
  hist->GetYaxis()->SetTitleFont(42);  
  hist->GetXaxis()->SetTitleSize(0.07);
  hist->GetYaxis()->SetTitleSize(0.07);
  hist->GetXaxis()->SetTitleOffset(0.9);
  hist->GetYaxis()->SetTitleOffset(1.3);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelSize(.07);
  hist->GetXaxis()->SetLabelSize(.07);
}

/*--------------------------------------------------------------------*/
void MakeNiceMapStyle(TH2 *hist)
/*--------------------------------------------------------------------*/
{
  hist->SetStats(kFALSE);  
  hist->GetXaxis()->CenterTitle(true);
  hist->GetYaxis()->CenterTitle(true);
  hist->GetZaxis()->CenterTitle(true);
  hist->GetXaxis()->SetTitleFont(42); 
  hist->GetYaxis()->SetTitleFont(42);
  hist->GetXaxis()->LabelsOption("v");
  hist->GetZaxis()->SetTitleFont(42);  
  hist->GetXaxis()->SetTitleSize(0.06);
  hist->GetYaxis()->SetTitleSize(0.06);
  hist->GetZaxis()->SetTitleSize(0.06);
  hist->GetXaxis()->SetTitleOffset(1.1);
  hist->GetZaxis()->SetTitleOffset(1.1);
  hist->GetYaxis()->SetTitleOffset(1.0);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  hist->GetZaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelSize(.05);
  hist->GetXaxis()->SetLabelSize(.05);
  hist->GetZaxis()->SetLabelSize(.05);
  
  Int_t nXCells = hist->GetNbinsX(); 
  Int_t nYCells = hist->GetNbinsY();
  Int_t nCells = nXCells*nYCells;

  Double_t min = 9999.; 
  Double_t max = -9999.;

  for(Int_t nX=1;nX<=nXCells;nX++){
    for(Int_t nY=1;nY<=nYCells;nY++){
      Double_t binContent = hist->GetBinContent(nX,nY);
      if(binContent>max) max = binContent;
      if(binContent<min) min = binContent;
    }
  }
  
  TH1F *histContentByCell = new TH1F(Form("histContentByCell_%s",hist->GetName()),"histContentByCell",nCells,min,max);

  for(Int_t nX=1;nX<=nXCells;nX++){
    for(Int_t nY=1;nY<=nYCells;nY++){
      histContentByCell->Fill(hist->GetBinContent(nX,nY));
    }
  }

  Double_t theMeanOfCells = histContentByCell->GetMean();
  Double_t theRMSOfCells  = histContentByCell->GetRMS();
  std::pair<Double_t,Double_t> theMAD = getMAD(histContentByCell);

 
  std::cout<<std::setw(24)<< left << hist->GetName() << "| mean: "<<std::setw(10)<<theMeanOfCells<<"| min: "<<std::setw(10)<< min <<"| max: "<<std::setw(10)<<max<<"| rms: "<<std::setw(10)<<theRMSOfCells<<"| mad: "<<std::setw(10)<<theMAD.first<<std::endl;

  TCanvas *cCheck = new TCanvas(Form("cCheck_%s",hist->GetName()),Form("cCheck_%s",hist->GetName()),800,800);
  cCheck->Divide(1,2);
  cCheck->cd(1);
  hist->Draw("box");
  cCheck->cd(2)->SetLogy();

  histContentByCell->Draw();
  
  //Double_t theNewMin = theMeanOfCells-theRMSOfCells;
  //Double_t theNewMax = theMeanOfCells+theRMSOfCells;

  Double_t theNewMin = theMeanOfCells-theMAD.first*3;
  Double_t theNewMax = theMeanOfCells+theMAD.first*3;
  
  TArrow *l0 = new TArrow(theMeanOfCells,cCheck->GetUymin(),theMeanOfCells,histContentByCell->GetMaximum(),0.3,"|>");
  l0->SetAngle(60);
  l0->SetLineColor(kRed);
  l0->SetLineWidth(4);
  l0->Draw("same");

  TArrow *l1 = new TArrow(theNewMin,cCheck->GetUymin(),theNewMin,histContentByCell->GetMaximum(),0.3,"|>");
  l1->SetAngle(60);
  l1->SetLineColor(kBlue);
  l1->SetLineWidth(4);
  l1->Draw("same");
  
  TArrow *l2 = new TArrow(theNewMax,cCheck->GetUymin(),theNewMax,histContentByCell->GetMaximum(),0.3,"|>");
  l2->SetAngle(60);
  l2->SetLineColor(kBlue);
  l2->SetLineWidth(4);
  l2->Draw("same");

  for(Int_t nX=1;nX<=nXCells;nX++){
    for(Int_t nY=1;nY<=nYCells;nY++){
      Double_t binContent = hist->GetBinContent(nX,nY);
      if (binContent<=theNewMin) hist->SetBinContent(nX,nY,theNewMin);
      else if (binContent>=theNewMax) hist->SetBinContent(nX,nY,theNewMax);
    }
  }
  
  //delete histContentByCell;

  hist->GetZaxis()->SetRangeUser(0.99*theNewMin,0.99*theNewMax);

}

/*--------------------------------------------------------------------*/
void setStyle(){
/*--------------------------------------------------------------------*/

  writeExtraText = true;       // if extra text
  lumi_13TeV     = "p-p collisions";
  
  TH1::StatOverflows(kTRUE);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat("e");
  //gStyle->SetPadTopMargin(0.05);
  //gStyle->SetPadBottomMargin(0.15);
  //gStyle->SetPadLeftMargin(0.17);
  //gStyle->SetPadRightMargin(0.02);
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

  const Int_t NRGBs = 5;
  const Int_t NCont = 255;
  
  Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
  Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
  Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
  Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
  TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
  gStyle->SetNumberContours(NCont);

}

/*--------------------------------------------------------------------*/
TH1F* DrawZero(TH1F *hist,Int_t nbins,Double_t lowedge,Double_t highedge,Int_t iter)
/*--------------------------------------------------------------------*/
{ 

  TH1F *hzero = new TH1F(Form("hzero_%s_%i",hist->GetName(),iter),Form("hzero_%s_%i",hist->GetName(),iter),nbins,lowedge,highedge);
  for (Int_t i=0;i<hzero->GetNbinsX();i++){
    hzero->SetBinContent(i,0.);
    hzero->SetBinError(i,0.);
  }
  hzero->SetLineWidth(2);
  hzero->SetLineStyle(9);
  hzero->SetLineColor(kMagenta);
  
  return hzero;
}

/*--------------------------------------------------------------------*/
void makeNewXAxis (TH1F *h)
/*--------------------------------------------------------------------*/
{
  
  TString myTitle = h->GetName();
  float axmin = -999;
  float axmax = 999.;
  int ndiv = 510;
  if(myTitle.Contains("eta")){
    axmin = -2.5;
    axmax = 2.5;
    ndiv = 505;
  } else if (myTitle.Contains("phi")){
    axmin = -TMath::Pi();
    axmax = TMath::Pi();
    ndiv = 510;
  } else  {
    std::cout<<"unrecognized variable"<<std::endl;
  }
  
  // Remove the current axis
  h->GetXaxis()->SetLabelOffset(999);
  h->GetXaxis()->SetTickLength(0);
  
   // Redraw the new axis
  gPad->Update();
  
  TGaxis *newaxis = new TGaxis(gPad->GetUxmin(),gPad->GetUymin(),
			       gPad->GetUxmax(),gPad->GetUymin(),
			       axmin,
			       axmax,
			       ndiv,"SDH");
  
  TGaxis *newaxisup =  new TGaxis(gPad->GetUxmin(),gPad->GetUymax(),
                                  gPad->GetUxmax(),gPad->GetUymax(),
                                  axmin,
                                  axmax,                          
                                  ndiv,"-SDH");
    
  newaxis->SetLabelOffset(0.02);
  newaxis->SetLabelFont(42);
  newaxis->SetLabelSize(0.05);
  
  newaxisup->SetLabelOffset(-0.02);
  newaxisup->SetLabelFont(42);
  newaxisup->SetLabelSize(0);
  
  newaxis->Draw();
  newaxisup->Draw();

}


/*--------------------------------------------------------------------*/
void makeNewPairOfAxes (TH2F *h)
/*--------------------------------------------------------------------*/
{
  
  int ndivx = 505;
  float axmin = -2.5;
  float axmax = 2.5;

  int ndivy = 510;
  float aymin = -TMath::Pi();
  float aymax = TMath::Pi();

  // Remove the current axis
  h->GetXaxis()->SetLabelOffset(999);
  h->GetXaxis()->SetTickLength(0);

  h->GetYaxis()->SetLabelOffset(999);
  h->GetYaxis()->SetTickLength(0);
  
   // Redraw the new axis
  gPad->Update();
  
  TGaxis *newXaxis = new TGaxis(gPad->GetUxmin(),gPad->GetUymin(),
				gPad->GetUxmax(),gPad->GetUymin(),
				axmin,
				axmax,
				ndivx,"SDH");
  
  TGaxis *newXaxisup =  new TGaxis(gPad->GetUxmin(),gPad->GetUymax(),
				   gPad->GetUxmax(),gPad->GetUymax(),
				   axmin,
				   axmax,                          
				   ndivx,"-SDH");


  TGaxis *newYaxisR = new TGaxis(gPad->GetUxmin(),gPad->GetUymin(),
				 gPad->GetUxmin(),gPad->GetUymax(),
				 aymin,
				 aymax,
				 ndivy,"SDH");
  
  TGaxis *newYaxisL =  new TGaxis(gPad->GetUxmax(),gPad->GetUymin(),
				  gPad->GetUxmax(),gPad->GetUymax(),
				  aymin,
				  aymax,                          
				  ndivy,"-SDH");
    
  newXaxis->SetLabelOffset(0.02);
  newXaxis->SetLabelFont(42);
  newXaxis->SetLabelSize(0.055);
  
  newXaxisup->SetLabelOffset(-0.02);
  newXaxisup->SetLabelFont(42);
  newXaxisup->SetLabelSize(0);
  
  newXaxis->Draw();
  newXaxisup->Draw();

  newYaxisR->SetLabelOffset(0.02);
  newYaxisR->SetLabelFont(42);
  newYaxisR->SetLabelSize(0.055);
  
  newYaxisL->SetLabelOffset(-0.02);
  newYaxisL->SetLabelFont(42);
  newYaxisL->SetLabelSize(0);
  
  newYaxisR->Draw();
  newYaxisL->Draw();

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
    //cout<<"FitPVResiduals() fit Up done!"<<endl;
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
    // cout<<"FitPVResiduals() fit Down done!"<<endl;
  } 
}

/*--------------------------------------------------------------------*/
void MakeNiceTF1Style(TF1 *f1,Int_t color)
/*--------------------------------------------------------------------*/
{
  f1->SetLineColor(color);
  f1->SetLineWidth(3);
  f1->SetLineStyle(2);
}

/*--------------------------------------------------------------------*/
std::pair<Double_t,Double_t> getTheRangeUser(TH1F* thePlot, Limits* lims)
/*--------------------------------------------------------------------*/
{
  TString theTitle = thePlot->GetName();
  theTitle.ToLower();

  /*
    Double_t m_dxyPhiMax     = 40;
    Double_t m_dzPhiMax      = 40;
    Double_t m_dxyEtaMax     = 40;
    Double_t m_dzEtaMax      = 40;
    
    Double_t m_dxyPhiNormMax = 0.5;
    Double_t m_dzPhiNormMax  = 0.5;
    Double_t m_dxyEtaNormMax = 0.5;
    Double_t m_dzEtaNormMax  = 0.5;
    
    Double_t w_dxyPhiMax     = 150;
    Double_t w_dzPhiMax      = 150;
    Double_t w_dxyEtaMax     = 150;
    Double_t w_dzEtaMax      = 1000;
    
    Double_t w_dxyPhiNormMax = 1.8;
    Double_t w_dzPhiNormMax  = 1.8;
    Double_t w_dxyEtaNormMax = 1.8;
    Double_t w_dzEtaNormMax  = 1.8;   
  */

  std::pair<Double_t,Double_t> result;
  
  if (theTitle.Contains("norm")){
    if (theTitle.Contains("means")){
      if(theTitle.Contains("dxy")){
	if(theTitle.Contains("phi")){
	  result = std::make_pair(-lims->get_dxyPhiNormMax().first,lims->get_dxyPhiNormMax().first);
	} else if (theTitle.Contains("eta")){
	  result = std::make_pair(-lims->get_dxyEtaNormMax().first,lims->get_dxyEtaNormMax().first);
	}
      } else if(theTitle.Contains("dz")){
	if(theTitle.Contains("phi")){
	  result = std::make_pair(-lims->get_dzPhiNormMax().first,lims->get_dzPhiNormMax().first);
	} else if (theTitle.Contains("eta")){
	  result = std::make_pair(-lims->get_dzEtaNormMax().first,lims->get_dzEtaNormMax().first);
	}
      }
    } else if (theTitle.Contains("widths")){
      if(theTitle.Contains("dxy")){
	if(theTitle.Contains("phi")){
	  result = std::make_pair(lims->get_dxyPhiNormMax().second-1,lims->get_dxyPhiNormMax().second);
	} else if (theTitle.Contains("eta")){
	  result = std::make_pair(lims->get_dxyEtaNormMax().second-1,lims->get_dxyEtaNormMax().second);
	}
      } else if(theTitle.Contains("dz")){	
	if(theTitle.Contains("phi")){
	  result = std::make_pair(lims->get_dzPhiNormMax().second-1,lims->get_dzPhiNormMax().second);
	} else if (theTitle.Contains("eta")){
	  result = std::make_pair(lims->get_dzEtaNormMax().second-1,lims->get_dzEtaNormMax().second);
	}
      }
    } 
  } else {
    if (theTitle.Contains("means")){
      if(theTitle.Contains("dxy")){
	if(theTitle.Contains("phi")){
	  result = std::make_pair(-lims->get_dxyPhiMax().first,lims->get_dxyPhiMax().first);
	} else if (theTitle.Contains("eta")){
	  result = std::make_pair(-lims->get_dxyEtaMax().first,lims->get_dxyEtaMax().first);
	}
      } else if(theTitle.Contains("dz")){
	if(theTitle.Contains("phi")){
	  result = std::make_pair(-lims->get_dzPhiMax().first,lims->get_dzPhiMax().first);
	} else if (theTitle.Contains("eta")){
	  result = std::make_pair(-lims->get_dzEtaMax().first,lims->get_dzEtaMax().first);
	}
      }
    } else if (theTitle.Contains("widths")){
      if(theTitle.Contains("dxy")){
	if(theTitle.Contains("phi")){
	  result = std::make_pair(0.,lims->get_dxyPhiMax().second);
	} else if (theTitle.Contains("eta")){
	  result = std::make_pair(0.,lims->get_dxyEtaMax().second);
	}
      } else if(theTitle.Contains("dz")){	
	if(theTitle.Contains("phi")){
	  result = std::make_pair(0.,lims->get_dzPhiMax().second);
	} else if (theTitle.Contains("eta")){
	  result = std::make_pair(0.,lims->get_dzEtaMax().second);
	}
      }
    }
  }

  return result;
  
}
