/** \file
   implementation of class FedRawData

   \author Stefano ARGIRO
   \date 28 Jun 2005
*/

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <iostream>

using namespace std;

FEDRawData::FEDRawData() {}

FEDRawData::FEDRawData(size_t newsize, size_t wordsize) : data_(newsize) {
  if (newsize % wordsize != 0)
    throw cms::Exception("DataCorrupt") << "FEDRawData::resize: " << newsize << " is not a multiple of " << wordsize
                                        << " bytes." << endl;
}

FEDRawData::FEDRawData(const FEDRawData &in) : data_(in.data_) {}
FEDRawData::~FEDRawData() {}
const unsigned char *FEDRawData::data() const { return data_.data(); }

unsigned char *FEDRawData::data() { return data_.data(); }

void FEDRawData::resize(size_t newsize, size_t wordsize) {
  if (size() == newsize)
    return;

  data_.resize(newsize);

  if (newsize % wordsize != 0)
    throw cms::Exception("DataCorrupt") << "FEDRawData::resize: " << newsize << " is not a multiple of " << wordsize
                                        << " bytes." << endl;
}
