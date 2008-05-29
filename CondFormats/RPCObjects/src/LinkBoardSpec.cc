
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include <sstream>

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

std::string LinkBoardSpec::print(int depth ) const 
{
  std::ostringstream str;
  std::string type = (theMaster) ? "master" : "slave";
  str <<" LinkBoardSpec: " << std::endl
            <<" --->" <<type<<" linkBoardNumInLink: " << theLinkBoardNumInLink 
            << std::endl;
  depth--;
  if (depth >=0) {
    typedef std::vector<FebConnectorSpec>::const_iterator IT;
    for (IT it=theFebs.begin(); it != theFebs.end(); it++) str << (*it).print(depth);
  }
  return str.str();
}

