/**
   \file
   implementation of class FedRawData

   \author Stefano ARGIRO
   \version $Id: FEDRawData.cc,v 1.1 2005/07/06 16:37:54 argiro Exp $
   \date 28 Jun 2005
*/

#include <DataFormats/FEDRawData/interface/FEDRawData.h>



using namespace raw;

FEDRawData::FEDRawData(){}

FEDRawData::FEDRawData(size_t size):data_(size){}

const unsigned char * FEDRawData::data()const {return &data_[0];}

unsigned char * FEDRawData::data() {return &data_[0];}

void FEDRawData::resize(size_t size) {
  data_.resize(size);
}
