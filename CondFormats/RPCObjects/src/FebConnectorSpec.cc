#include "CondFormats/RPCObjects/interface/FebConnectorSpec.h"
#include "CondFormats/RPCObjects/interface/DBSpecToDetUnit.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <iostream>

FebConnectorSpec::FebConnectorSpec(
    int num, const ChamberLocationSpec & chamber, const FebLocationSpec & feb)
  : theLinkBoardInputNum(num),
    theChamber(chamber),
    theFeb(feb),
    theRawId(0)
{ }

void FebConnectorSpec::add(const ChamberStripSpec & strip)
{
  theStrips.push_back(strip);
}

const ChamberStripSpec * FebConnectorSpec::strip( int pinNumber) const
{
  //FIXME - temporary implementaion, to be replace by LUT (in preparation)
  typedef std::vector<ChamberStripSpec>::const_iterator IT;
  for (IT it=theStrips.begin(); it != theStrips.end(); it++) {
    if(pinNumber==it->cablePinNumber) return &(*it);
  }
  return 0;
}

const uint32_t & FebConnectorSpec::rawId() const
{
  DBSpecToDetUnit toDU;
  if (!theRawId)  theRawId = toDU(theChamber, theFeb);
  return theRawId;
}


std::string FebConnectorSpec::print(int depth) const
{
  std::ostringstream str;
  str << "FebConnectorSpec in LinkBoardNum ="<<linkBoardInputNum()
      <<" rawId: " << rawId() <<std::endl;
  //RPCDetId aDet(rawId());
  //std::cout<<aDet;
  str << theChamber.print(depth)<< std::endl << theFeb.print(depth) ;
  depth--;
  if (depth >=0) {
    typedef std::vector<ChamberStripSpec>::const_iterator IT;
    for (IT it=theStrips.begin(); it != theStrips.end(); it++) str << (*it).print(depth); 
  }
  return str.str();
}
