/**
   \file
   implementation of class FedRawData

   \author Stefano ARGIRO
   \version $Id$
   \date 28 Jun 2005
*/

static const char CVSId[] = "$Id$";


#include <DataFormats/FEDRawData/interface/FEDRawData.h>



using namespace raw;

FEDRawData::FEDRawData(){}

FEDRawData::FEDRawData(size_t size):data_(size){}

const unsigned char * FEDRawData::data()const {return &data_[0];}

