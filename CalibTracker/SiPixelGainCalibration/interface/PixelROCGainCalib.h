//---------------------------------------------------

// Author : Freya.Blekman@cern.ch
// Name   : PixelROCGainCalib

//---------------------------------------------------

#ifndef PixelROCGainCalib_h
#define PixelROCGainCalib_h

#include "TH1F.h"
#include "TH2F.h"
#include "TObjArray.h"
#include "TF1.h"
#include "TObject.h"
#include <iostream>
#include <map>
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibPixel.h"

class PixelROCGainCalib{
 public:
  PixelROCGainCalib();
  ~PixelROCGainCalib() {;}

  void init(unsigned int linkid, unsigned int rocid,unsigned int nvcal,unsigned int vcalRangeMin, unsigned vcalRangeMax, unsigned vcalRangeStep,unsigned int ncols, unsigned int nrows);

  TH1F *gethisto(unsigned int row, unsigned int col);
  //  TH1F * addHistoFileService(TFileDirectory dir,unsigned int row, unsigned int col);
  void fill(unsigned int row,unsigned int col,unsigned int vcal,unsigned int adc);
  void fit(unsigned int row,unsigned int col, TF1* function);
  bool isvalid(unsigned int row,unsigned int col) {return pixelUsed_[row][col];}
  TString getTitle(){return thisROCTitle_;}
  unsigned int getDetID() {return detid_;}
  void setDetID(unsigned int val){detid_=val;}
  //void setVCalRange(unsigned int vcalRangeMin, unsigned vcalRangeMax, unsigned vcalRangeStep){ vcalrangemin_=vcalRangeMin;vcalrangemax_=vcalRangeMax; vcalrangestep_=vcalRangeStep;}
  
 private:
  PixelROCGainCalibPixel thePixels_[80][52];// [rows][colums]
  bool pixelUsed_[80][52];// [rows][columns]
  unsigned int vcalrangestep_; 
  unsigned int vcalrangemin_;
  unsigned int vcalrangemax_;
  unsigned int nrowsmax_;
  unsigned int nrowslimit_;
  unsigned int ncolsmax_;
  unsigned int ncolslimit_;
  unsigned int linkid_;
  unsigned int rocid_;
  unsigned int nvcal_;
  unsigned int detid_;
  bool checkRowCols(unsigned int row, unsigned int col);
  TString createTitle(unsigned int row, unsigned int col);
  TString thisROCTitle_;
  
  std::map < unsigned int , unsigned int > vcalmap_;// keeps track of all used vcals
}; 


#endif
