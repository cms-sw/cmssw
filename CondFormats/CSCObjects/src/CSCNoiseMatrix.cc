#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <sstream>

CSCNoiseMatrix::CSCNoiseMatrix(){}
CSCNoiseMatrix::~CSCNoiseMatrix(){}

const CSCNoiseMatrix::Item & CSCNoiseMatrix::item(const CSCDetId & cscId, int strip) const
{
  CSCIndexer indexer;
  NoiseMatrixMap::const_iterator mapItr = matrix.find( indexer.dbIndex(cscId, strip) );
  if(mapItr == matrix.end())
  {
    throw cms::Exception("CSCNoiseMatrix")
      << "Cannot find CSC conditions for chamber " << CSCDetId(cscId);
  }
  return mapItr->second.at(strip-1);
}


std::string CSCNoiseMatrix::Item::print() const
{
  std::ostringstream os;
  os << elem33 << " " << elem34 << " " << elem35 << " " << elem44 << " "
     << elem45 << " " << elem46 << " " << elem55 << " " << elem56 << " "
     << elem57 << " " << elem66 << " " << elem67 << " " << elem77 << "\n";
  return os.str();
}

std::string CSCNoiseMatrix::print() const
{
  std::ostringstream os;
  for(NoiseMatrixMap::const_iterator mapItr = matrix.begin(); mapItr != matrix.end(); ++mapItr)
  {
    os << mapItr->first<< " ";
    for(std::vector<Item>::const_iterator itemItr = mapItr->second.begin(); 
        itemItr != mapItr->second.end(); ++itemItr)
    {
      os << itemItr->print();
    }
  }
  return os.str();
}

