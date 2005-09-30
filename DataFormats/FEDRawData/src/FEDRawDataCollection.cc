/** \file DaqRawDataCollection.cc
 *  implementation of DaqRawDataCollection
 *
 *  $Date: 2005/07/06 16:37:54 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - S. Argiro'
 */


#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include <iostream>

using namespace raw;
using namespace std;

FEDRawDataCollection::FEDRawDataCollection():
  data_(FEDNumbering::lastFEDId()+1) {}


FEDRawDataCollection::~FEDRawDataCollection(){}


const FEDRawData&   FEDRawDataCollection::FEDData(int fedid) const {
  return data_[fedid];
}


FEDRawData&   FEDRawDataCollection::FEDData(int fedid) {
  return data_[fedid];
}
