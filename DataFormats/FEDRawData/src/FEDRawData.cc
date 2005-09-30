/**
   \file
   implementation of class FedRawData

   \author Stefano ARGIRO
   \version $Id: FEDRawData.cc,v 1.2 2005/09/30 08:12:56 namapane Exp $
   \date 28 Jun 2005
*/

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <iostream>

using namespace raw;
using namespace std;

FEDRawData::FEDRawData(){}

FEDRawData::FEDRawData(size_t size):data_(size){}

const unsigned char * FEDRawData::data()const {return &data_[0];}

unsigned char * FEDRawData::data() {return &data_[0];}

void FEDRawData::resize(size_t size) {
  if (size%8!=0) throw cms::Exception("DataCorrupt") << "FEDRawData::resize: " << size << " is not a multiple of 8 bytes." << endl;
  data_.resize(size);
}
