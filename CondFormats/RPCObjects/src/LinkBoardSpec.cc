#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include <iostream>

LinkBoardSpec::LinkBoardSpec(bool m, int l)
    : theMaster(m), theLinkBoardNumInLink(l) { }

void LinkBoardSpec::add(const FebConnectorSpec & feb)
{
  theFebs.push_back(feb);
}

const FebConnectorSpec * LinkBoardSpec::feb(int febInputNum) const
{
  //FIXME - temporary implementaion, to be replace by LUT (in preparation)
  typedef std::vector<FebConnectorSpec>::const_iterator IT;
  for (IT it=theFebs.begin(); it != theFebs.end(); it++) {
    if(febInputNum==it->linkBoardInputNum()) return &(*it);
  }
  return 0;
}

void LinkBoardSpec::print(int depth ) const 
{
  if (depth<0) return;
  depth--;
  std::string type = (theMaster) ? "master" : "slave";
  std::cout <<" LinkBoardSpec: " << std::endl
            <<" --->" <<type<<" linkBoardNumInLink: " << theLinkBoardNumInLink 
            << std::endl;
  typedef std::vector<FebConnectorSpec>::const_iterator IT;
  for (IT it=theFebs.begin(); it != theFebs.end(); it++) (*it).print(depth);
}


