//---------------------------------------------------

// Author : Freya.Blekman@cern.ch
// Name   : PixelROCGainCalib

//---------------------------------------------------

#ifndef PixelROCGainCalib_h
#define PixelROCGainCalib_h

#include "TH1F.h"
#include "TH2F.h"
#include <vector>
#include "TF1.h"
#include "TObject.h"
#include <iostream>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "PhysicsTools/UtilAlgos/interface/TFileDirectory.h"

#include <map>
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibPixel.h"

class PixelROCGainCalib{
 public:
  PixelROCGainCalib(uint32_t maxrow, uint32_t maxcol, uint32_t nvcol);
  ~PixelROCGainCalib();
  
  void printsummary(uint32_t row, uint32_t col);

  void init(std::string name, uint32_t detid,uint32_t nvcal,uint32_t vcalRangeMin, uint32_t vcalRangeMax, unsigned vcalRangeStep,uint32_t ncols, uint32_t nrows, uint32_t ntriggers, edm::Service<TFileService>  therootfileservice);

  //  TGraph *getgraph(uint32_t row, uint32_t col);
  TH1F *gethisto(uint32_t row, uint32_t col){edm::Service<TFileService> afileservice; return gethisto(row,col,afileservice);}
  TH1F *gethisto(uint32_t row, uint32_t col, edm::Service<TFileService> therootfileservice);
  
  void fill(uint32_t row,uint32_t col,uint32_t ipoint,uint32_t adc,bool verbose=false);
  void fillVcal(uint32_t row,uint32_t col,uint32_t vcal,uint32_t adc,bool verbose=false);
  std::string getTitle(){return thisROCTitle_;}
  uint32_t getDetID() {return detid_;}

  bool isvalid(uint32_t row, uint32_t col);
  bool isfilled(uint32_t row, uint32_t col);

  uint32_t getVcalBin(uint32_t vcalval){ return (vcalval - vcalrangemin_)/vcalrangestep_ ;}
  uint32_t getBinVcal(uint32_t ibin) {return vcalrangemin_ +(vcalrangestep_*ibin);}
  uint32_t getNcols() {return ncolslimit_;}
  uint32_t getNrows() {return nrowslimit_;}
  uint32_t getNentries() {return nentries_;}
 private:
  std::string createTitle(uint32_t row, uint32_t col);

  // class members:
  //  std::vector< std::vector < TH1F* > > thePixels_;  
  std::vector < std::vector < PixelROCGainCalibPixel > > thePixels_;
  std::vector<uint32_t> points;
  
  uint32_t nentries_;
  uint32_t vcalrangestep_;
  uint32_t vcalrangemin_;
  uint32_t vcalrangemax_;
  uint32_t nvcal_;
  uint32_t ncolslimit_;
  uint32_t nrowslimit_;
  std::string thisROCTitle_;
  uint32_t detid_;
  uint32_t ntriggers_;
  
};

inline bool PixelROCGainCalib::isvalid(uint32_t row, uint32_t column){
  
  assert(ncolslimit_>column);
  assert(nrowslimit_>row);
}

#endif
