/** \file
 *  implementation of DaqRawDataCollection
 *
 *  \author N. Amapane - S. Argiro'
 */

#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include <iostream>

using namespace std;

FEDRawDataCollection::FEDRawDataCollection() : data_(FEDNumbering::lastFEDId() + 1) {}
