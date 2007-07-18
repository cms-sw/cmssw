#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCNoiseMatrix::CSCNoiseMatrix(){}
CSCNoiseMatrix::~CSCNoiseMatrix(){}

const CSCNoiseMatrix::Item & CSCNoiseMatrix::item(int cscId, int strip) const
{
  NoiseMatrixMap::const_iterator mapItr = matrix.find(cscId);
  if(mapItr == matrix.end())
  {
    throw cms::Exception("CSCNoiseMatrix")
      << "Cannot find CSC conditions for chamber " << cscId;
  }
  return mapItr->second.at(strip-1);
}

