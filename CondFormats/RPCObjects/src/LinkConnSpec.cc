#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"
#include <iostream>

void LinkConnSpec::print(int depth) const
{
  if(depth<0) return;
  std::cout << "LinkConnSpec number="<<theTriggerBoardInputNumber<<std::endl;
  typedef std::vector<LinkBoardSpec>::const_iterator ILB;
  depth--;
  for (ILB it = theLBs.begin(); it != theLBs.end(); it++) (*it).print(depth);  
}

void LinkConnSpec::add(const LinkBoardSpec & lb) 
{ 
  theLBs.push_back(lb); 
}

const LinkBoardSpec * LinkConnSpec::linkBoard( int linkBoardNumInLink) const
{
  //FIXME - temporary implementaion, to be replace by LUT (in preparation)
  typedef std::vector<LinkBoardSpec>::const_iterator IT;
  for (IT it=theLBs.begin(); it != theLBs.end(); it++) {
    if(linkBoardNumInLink==it->linkBoardNumInLink()) return &(*it);
  }
  return 0;
}

