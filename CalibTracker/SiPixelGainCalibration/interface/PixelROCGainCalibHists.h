#ifndef PixelROCGainCalibHists_h
#define PixelROCGainCalibHists_h

#include "TH1F.h"
#include "TH2F.h"
#include "TObjArray.h"
#include <iostream>

class PixelROCGainCalibHists {
 public:
  
  PixelROCGainCalibHists();

  void init(unsigned int linkid, unsigned int rocid,unsigned int nvcal);

  void cleanup(void);// deletes everything
  bool filled(unsigned int row,unsigned int col);

  void draw(unsigned int row,unsigned int col);

  void save(unsigned int row,unsigned int col, TFile *rootfile); // save the appropriate histogram (with possible functions that come with it) to file.

  void fill(unsigned int row,unsigned int col,unsigned int vcal,unsigned int adc);
  TF1* fit(unsigned int row,unsigned int col);
  void setFitRange(unsigned int vcalmin, unsigned int vcalmax){vcalmin_=vcalmin;vcalmax_=vcalmax;}
  
  void setVCalRange(unsigned int vcalRangeMin, unsigned vcalRangeMax, unsigned vcalRangeStep){ vcalrangemin_=vcalRangeMin;vcalrangemax_=vcalRangeMax; vcalrangestep_=vcalRangeStep;}
 private:

  unsigned int vcalmin_;
  unsigned int vcalmax_;
  unsigned int nrowsmax_;
  unsigned int ncolsmax_;
  unsigned int linkid_;
  unsigned int rocid_;
  unsigned int nvcal_;
  unsigned int vcalrangemin_;
  unsigned int vcalrangemax_;
  unsigned int vcalrangestep_; 

  TString rootfilename_;
  bool anyhistofilled_;
  bool checkRowCols(unsigned int row, unsigned int col);
  
  TH1F* adc_hist[80][52]; 
  TH1F* adc_hist_nentries[80][52]; 
  // TGraphErrors *adc_graph[80][52]; // graphs might be better as there is more control on the size of the object... to be investigated.
  
  TH2F *overview_adc_hist;
  void fixHistogram1D(TH1F *histo,TString xtitle, TString ytitle, int colour);
  std::string histofilename_;
  TF1 *thefitfunction_;
  TString functionname_;
  
};

#endif
