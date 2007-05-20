#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibHists.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TH1F.h"
#include "TGraphErrors.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TStyle.h"
#include "TMath.h"
#include "TSystem.h"
#include <iostream>


PixelROCGainCalibHists::PixelROCGainCalibHists():vcalmin_(0),vcalmax_(256),nrowsmax_(80),ncolsmax_(52),nvcal_(1),vcalrangemin_(0),vcalrangemax_(256),vcalrangestep_(256),functionname_("thefitfunction"){
  // edm::LogVerbatim("") << "In PixelROCGainCalibHists constructor, using " << nrowsmax_ << " rows and " << ncolsmax_ << "columns." <<std::endl;
  
  

}

void PixelROCGainCalibHists::save(unsigned int row,unsigned int col, TFile *rootfile){
  if(!filled(row,col))
    return;
 //  if(!rootfile->FindObject("PixelHistograms"))
//     rootfile->mkdir("PixelHistograms");
//   rootfile->cd("PixelHistograms");
  rootfile->cd();
  TH1F *histo =  adc_hist[row][col];
  edm::LogVerbatim("") << "Saving histogram " << histo->GetName() << std::endl;
  histo->Write(histo->GetName());
  rootfile->cd();
//   if( histo->GetFunction(functionname_)){
//     thefitfunction_ = histo->GetFunction(functionname_);
//     TString funcname =histo->GetName();
//     funcname+=" fit function";
//     thefitfunction_->Write(funcname);
//   edm::LogVerbatim("") << "Saving function " << thefitfunction_->GetName() << std::endl;
//   }
}
void PixelROCGainCalibHists::init(unsigned int linkid, unsigned int rocid,unsigned int nvcal){
  linkid_=linkid;
  rocid_=rocid;
  nvcal_=nvcal;
  anyhistofilled_=false;
  gStyle->SetOptStat(0);
  for(unsigned int row=0;row<nrowsmax_;row++){
    for(unsigned int col=0;col<ncolsmax_;col++){
      adc_hist[row][col]=0;
      adc_hist_nentries[row][col]=0;
      //      adc_graph[row][col]=0;
    }
  }
  thefitfunction_ = new TF1(functionname_,"pol1",20,100);//(double)vcalmin_,(double)vcalmax_);
  return;
}

bool PixelROCGainCalibHists::filled(unsigned int row,unsigned int col){
  if(!checkRowCols(row,col)){
    edm::LogVerbatim("") << "PixelROCGainCalibHists::filled() WARNING, column or row out of range" << std::endl;
    return false;
  }
  if(adc_hist[row][col]==0)
    return false;
  else if(adc_hist[row][col]->GetEntries()>0)
    return true;
  else
    return false;
}



void PixelROCGainCalibHists::draw(unsigned int row,unsigned int col){
  if(!checkRowCols(row,col)){
    edm::LogVerbatim("") << "PixelROCGainCalibHists::draw() WARNING, column or row out of range" << std::endl;
    return ;
  }
  //  TCanvas *mycanv = new TCanvas("mycanv","mycanv");
  adc_hist[row][col]->Draw();
}


void PixelROCGainCalibHists::fill(unsigned int row,unsigned int col,unsigned int vcal,unsigned int adc){
  //  edm::LogVerbatim("") << "filling histogram " << row << "," << col << " with VCAL,ADC: " << vcal << "," << adc << std::endl; 
  anyhistofilled_=true;
  if(!checkRowCols(row,col)){
    edm::LogVerbatim("") << "PixelROCGainCalibHists::fill() WARNING, column or row out of range" << std::endl;
    return ;
  }
  TH1F *hist=adc_hist[row][col];
  TH1F *hist2 = adc_hist_nentries[row][col];
  if (adc_hist[row][col]==0) {
    
    TString name="Channel=";
    name+=(linkid_);
    name=name+" ROC=";
    name+=(rocid_);
    name=name+" row=";
    name+=(row);
    name=name+" col=";
    name+=(col);
    //    edm::LogVerbatim("") << "creating histogram: " << name << std::endl;
    
    float startval = vcalrangemin_;
    float endval = vcalrangemax_;
    hist=adc_hist[row][col]=new TH1F(name,name,nvcal_+1,startval,endval);
    name+=", nentries";
    hist2=adc_hist_nentries[row][col]=new TH1F(name,name,nvcal_+1,startval,endval);
    hist->Sumw2();
    //    hist2->Sumw2();
    fixHistogram1D(hist,"VCAL input","response [ADC counts]",1);
    
    //    adc_graph[row][col]= new TGraphErrors(name,nvcal,"","");
  }
  hist->Fill(vcal,adc);
  hist2->Fill(vcal,1.);
  //edm::LogVerbatim("") << "number of entries is: " << hist->GetEntries() << std::endl;
}
////////////////////////////
//   method that includes fit function
///////////////////////

TF1* PixelROCGainCalibHists::fit(unsigned int row,unsigned int col){
  if(!filled(row,col))
    return 0;
  TH1F *hist2 = adc_hist_nentries[row][col];
  if(!checkRowCols(row,col)){
    edm::LogVerbatim("") << "PixelROCGainCalibHists::fit() WARNING, column or row out of range" << std::endl;
    return 0;
  }
 
  TH1F *hist = adc_hist[row][col];
  hist->Divide(hist2);
  hist->SetMinimum(0.);
  hist->SetMaximum(256.0);
  thefitfunction_ ->SetParameter(0,100.);
  thefitfunction_ ->SetParameter(1,1.);
  //  thefitfunction_ ->Print();
  hist->Fit(thefitfunction_,"R");
  
  return thefitfunction_;
}

// check on the rows and columns
bool PixelROCGainCalibHists::checkRowCols(unsigned int row, unsigned int col){
  if(row>=nrowsmax_)
    return false;
  if(col>=ncolsmax_)
    return false;
  
  return true;
}
void PixelROCGainCalibHists::cleanup(void){
// manually delete some stuff
  if(overview_adc_hist) delete overview_adc_hist ;
  for(unsigned int irow = 0; irow < nrowsmax_; ++irow){
    for(unsigned int icol=0; icol < ncolsmax_; ++icol){
      if(adc_hist[irow][icol]!=0)
	delete adc_hist[irow][icol];
      adc_hist[irow][icol]=0;
      if(adc_hist_nentries[irow][icol]!=0)
	delete adc_hist_nentries[irow][icol];
      adc_hist_nentries[irow][icol]=0;
    }
  }
}
// added by Freya (who is into aesthetics :)
void PixelROCGainCalibHists::fixHistogram1D(TH1F *histo,TString xtitle, TString ytitle, int colour){

  histo->SetLineColor(colour);
  histo->SetMarkerColor(colour);
  histo->SetMarkerStyle(20);
  histo->SetMarkerSize(1.1);  
  histo->SetLineWidth(3);
  histo->GetXaxis()->SetTitle(xtitle);
  histo->GetYaxis()->SetTitle(ytitle);
  histo->GetXaxis()->SetTitleSize(0.05);
  histo->GetYaxis()->SetTitleSize(0.05);
  histo->GetXaxis()->SetTitleOffset(1.05);
  histo->GetYaxis()->SetTitleOffset(1.05);
  histo->GetXaxis()->SetTitleColor(1);
  histo->GetYaxis()->SetTitleColor(1);
  histo->GetXaxis()->SetNdivisions(10);
  histo->GetYaxis()->SetNdivisions(10); 
  histo->GetXaxis()->SetLabelFont(62);
  histo->GetYaxis()->SetLabelFont(62);
  histo->GetXaxis()->SetLabelOffset(0.004);
  histo->GetYaxis()->SetLabelOffset(0.004);
  histo->GetXaxis()->SetLabelSize(0.05);
  histo->GetYaxis()->SetLabelSize(0.045);
  histo->SetDrawOption("p");
  return;
}

