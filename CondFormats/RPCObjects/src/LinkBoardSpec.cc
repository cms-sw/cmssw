#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include <iostream>

LinkBoardSpec::LinkBoardSpec(bool m, int l, const ChamberLocationSpec & c)
    : theMaster(m), theLinkBoardNumInLink(l), theChamberSpec(c) { }

void LinkBoardSpec::add(const FebSpec & feb)
{
  theFebs.push_back(feb);
}

const FebSpec * LinkBoardSpec::feb(int febInputNum) const
{
  //FIXME - temporary implementaion, to be replace by LUT (in preparation)
  typedef std::vector<FebSpec>::const_iterator IT;
  for (IT it=theFebs.begin(); it != theFebs.end(); it++) {
    if(febInputNum==it->linkBoardInputNum()) return &(*it);
  }
  return 0;
}

void LinkBoardSpec::print(int depth ) const 
{
  if (depth<0) return;
  std::string type = (theMaster) ? "master" : "slave";
  std::cout <<" LinkBoardSpec: " << std::endl
            <<" --->" <<type<<" linkBoardNumInLink: " << theLinkBoardNumInLink 
            << std::endl;
  theChamberSpec.print(--depth);
}
