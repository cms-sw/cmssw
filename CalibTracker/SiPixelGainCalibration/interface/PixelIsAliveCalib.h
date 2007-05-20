#ifndef PixelIsAliveCalib_h
#define PixelIsAliveCalib_h

#include "TH1F.h"
#include "TH2F.h"
#include "TObjArray.h"
#include <iostream>

class PixelIsAliveCalib {
 public:
  
  PixelIsAliveCalib();

  void init(unsigned int linkid, unsigned int rocid,unsigned int nvcal);

  void cleanup(void);// deletes everything
  bool filled(unsigned int row,unsigned int col);

  void draw(unsigned int row,unsigned int col);

  void fill(unsigned int row,unsigned int col,unsigned int adc);
  TH2F *getResult(void);
  
 private:

  unsigned int nrowsmax_;
  unsigned int ncolsmax_;
  unsigned int linkid_;
  unsigned int rocid_;
  unsigned int nTriggersPerPixel_;

  bool anyhistofilled_;
  bool checkRowCols(unsigned int row, unsigned int col);
 
  TH2F *is_alive_hist;
  
  void fixHistogram2D(TH2F *histo,TString xtitle, TString ytitle, int colour);
  std::string histofilename_;
  
};

#endif
