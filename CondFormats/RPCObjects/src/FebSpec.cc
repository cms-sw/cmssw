#include "CondFormats/RPCObjects/interface/FebSpec.h"
#include "CondFormats/RPCObjects/interface/DBSpecToDetUnit.h"
#include <iostream>

void FebSpec::print(int depth) const
{
  if(depth<0) return;
  std::cout << "FebSpec in LinkBoardNum ="<<linkBoardInputNum()<<std::endl;
  depth--;
  typedef std::vector<ChamberStripSpec>::const_iterator IT;
  for (IT it=theStrips.begin(); it != theStrips.end(); it++) (*it).print(depth); 
  std::cout << std::endl;
}

void FebSpec::add(const ChamberStripSpec & strip) 
{ 
  theStrips.push_back(strip); 
}

const ChamberStripSpec * FebSpec::strip( int pinNumber) const
{
  //FIXME - temporary implementaion, to be replace by LUT (in preparation)
  typedef std::vector<ChamberStripSpec>::const_iterator IT;
  for (IT it=theStrips.begin(); it != theStrips.end(); it++) {
    if(pinNumber==it->cablePinNumber) return &(*it);
  }
  return 0;
}

const uint32_t & FebSpec::rawId(const ChamberLocationSpec & location) const
{
  DBSpecToDetUnit toDU;
  if (!theRawId) theRawId = toDU(location, febLocalEtaPartition()); 
  return theRawId;
}
