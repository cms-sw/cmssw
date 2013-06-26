/** \file
   implementation of class FedRawData

   \author Stefano ARGIRO
   \version $Id: FEDRawData.cc,v 1.6 2005/10/18 13:29:53 meschi Exp $
   \date 28 Jun 2005
*/

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <iostream>

using namespace std;

FEDRawData::FEDRawData()
{
}

FEDRawData::FEDRawData(size_t newsize):data_(newsize){
  if (newsize%8!=0) throw cms::Exception("DataCorrupt") << "FEDRawData::resize: " << newsize << " is not a multiple of 8 bytes." << endl;
}

FEDRawData::FEDRawData(const FEDRawData &in) : data_(in.data_)
{
}
FEDRawData::~FEDRawData()
{
}
const unsigned char * FEDRawData::data()const {return &data_[0];}

unsigned char * FEDRawData::data() {return &data_[0];}

void FEDRawData::resize(size_t newsize) {
  if (size()==newsize) return;

  data_.resize(newsize);

  if (newsize%8!=0) throw cms::Exception("DataCorrupt") << "FEDRawData::resize: " << newsize << " is not a multiple of 8 bytes." << endl;
}
