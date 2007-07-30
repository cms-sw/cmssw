#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

EcalDataFrame::EcalDataFrame() : size_(0),
				 data_(MAXSAMPLES)
{
}


void EcalDataFrame::setSize(const int& size) 
{
  if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else if (size<=0) size_=0;
  else size_=size;
}


  
