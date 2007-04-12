#ifndef PixelROCGainCalib_h
#define PixelROCGainCalib_h

#include "TH1F.h"
#include <iostream>

class PixelROCGainCalib {

 public:
  
  PixelROCGainCalib();

  void init(unsigned int linkid, unsigned int rocid,unsigned int nvcal);

  bool filled(unsigned int row,unsigned int col);

  void draw(unsigned int row,unsigned int col);

  void fill(unsigned int row,unsigned int col,unsigned int vcal,unsigned int adc);

 private:

  unsigned int linkid_;
  unsigned int rocid_;
  unsigned int nvcal_;

  TH1F* adc_hist[80][52];
  
};

#endif
