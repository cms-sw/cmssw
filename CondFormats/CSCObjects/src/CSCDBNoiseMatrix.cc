#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCDBNoiseMatrix::CSCDBNoiseMatrix(){}
CSCDBNoiseMatrix::~CSCDBNoiseMatrix(){}
/*
const CSCDBNoiseMatrix::Item & CSCDBNoiseMatrix::item(int cscId, int strip) const
{
  NoiseMatrixContainer::const_iterator Itr = matrix.find(cscId);
  if(Itr == matrix.end())
  {
    throw cms::Exception("CSCDBNoiseMatrix")
      << "Cannot find CSC conditions for chamber " << cscId;
  }
  return Itr->second.at(strip-1);
}

*/
