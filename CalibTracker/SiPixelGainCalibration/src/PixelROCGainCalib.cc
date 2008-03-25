#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalib.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
//---------------------------------------------------

// Author : Freya.Blekman@cern.ch
// Name   : PixelROCGainCalib

//---------------------------------------------------


PixelROCGainCalib::PixelROCGainCalib(): 
vcalrangemin_(0),vcalrangestep_(256),vcalrangemax_(256),nrowsmax_(80),ncolsmax_(52),
linkid_(0),rocid_(0),nvcal_(1), thisROCTitle_(""),ncolslimit_(52),nrowslimit_(80)
{
  // some ugly asserts to make sure none of the arrays run over limits 
}
//*********************
void PixelROCGainCalib::init(unsigned int linkid, unsigned int rocid,unsigned int nvcal,unsigned int vcalRangeMin, unsigned vcalRangeMax, unsigned vcalRangeStep,unsigned int ncols,unsigned int nrows){
  linkid_=linkid;
  rocid_=rocid;
  nvcal_=nvcal; 
  vcalrangemin_=vcalRangeMin;
  vcalrangemax_=vcalRangeMax; 
  vcalrangestep_=vcalRangeStep;
  nrowsmax_=nrows;
  ncolsmax_=ncols;
  thisROCTitle_="Channel_";
  thisROCTitle_+=linkid_;
  thisROCTitle_+="_ROC_";
  thisROCTitle_+=rocid_;

  for(unsigned int irow=0; irow<nrowsmax_;++irow){
    for(unsigned int icol=0; icol<ncolsmax_;++icol){
      pixelUsed_[irow][icol]=false;
    }
  }
  unsigned int iwork=0;
  for(unsigned int vcalwork = vcalrangemin_; vcalwork<=vcalrangemax_;vcalwork+=vcalrangestep_){
    vcalmap_[vcalwork]=iwork;
    iwork++;
  }
}


//**********************
void PixelROCGainCalib::fill(unsigned int row,unsigned int col,unsigned int vcal,unsigned int adc){
  if(!checkRowCols(row,col)){
    edm::LogVerbatim("WARNING") <<" PixelROCGainCalib::fill() WARNING, column or row out of range, value row: " << row << ", col:" << col << std::endl;
    return ;
  }
  if(!pixelUsed_[row][col]){
    // actually create the PixelROCGainCalibPixel object
    thePixels_[row][col].init(nvcal_);
   
    pixelUsed_[row][col]=true;
  }
  // now fill the object:
  //  std::cout <<"Filling PixelROCGainCalibObject "  <<  vcalmap_[vcal] << " " << vcal << std::endl;
  thePixels_[row][col].addPoint(vcalmap_[vcal],adc,vcal); 
  return;
}
//**********************
TH1F* PixelROCGainCalib::gethisto(unsigned int row, unsigned int col){
  if(!pixelUsed_[row][col])
    return 0;
  
  TH1F *result = (TH1F*) thePixels_[row][col].createHistogram(createTitle(row,col),createTitle(row,col),nvcal_,vcalrangemin_,vcalrangemax_);
  //  std::cout << createTitle(row,col) << " " << nvcal_ << " " << vcalrangemin_ << " " << vcalrangemax_ << std::endl;
  return result;
}
//**********************
// TH1F* PixelROCGainCalib::getHistoFileService(TFileDirectory dir,unsigned int row, unsigned int col){
//   if(!pixelUsed_[row][col])
//     return 0;
  
//   TH1F *result = (TH1F*) thePixels_[row][col].createHistogramFileService(dir,createTitle(row,col),createTitle(row,col),nvcal_,vcalrangemin_,vcalrangemax_);
//   //  std::cout << createTitle(row,col) << " " << nvcal_ << " " << vcalrangemin_ << " " << vcalrangemax_ << std::endl;
//   return result;
// }
//**********************
void PixelROCGainCalib::fit(unsigned int row,unsigned int col, TF1* function){
  gethisto(row,col)->Fit(function,"R");
  return;
}
//**********************
bool PixelROCGainCalib::checkRowCols(unsigned int row, unsigned int col){
  if(row>=nrowsmax_)
    return false;
  if(col>=ncolsmax_)
    return false;
  
  return true;
}
//**********************
TString PixelROCGainCalib::createTitle(unsigned int row, unsigned int col){
 
  TString result = thisROCTitle_;
  result+="_row_";
  result+=row;
  result+="_col_";
  result+=col;
  return result;
}
