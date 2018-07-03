#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"
#include <sstream>

std::string LinkConnSpec::print(int depth) const
{
  std::ostringstream str;
  str << "LinkConnSpec number="<<theTriggerBoardInputNumber<<std::endl;
  depth--;
  if (depth >= 0) {
    typedef std::vector<LinkBoardSpec>::const_iterator ILB;
    for (ILB it = theLBs.begin(); it != theLBs.end(); it++) str << (*it).print(depth);  
  }
  return str.str();
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
  return nullptr;
}

