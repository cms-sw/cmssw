#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"
#include <sstream>

std::string LinkConnSpec::print(int depth) const {
  std::ostringstream str;
  str << "LinkConnSpec number=" << theTriggerBoardInputNumber << std::endl;
  depth--;
  if (depth >= 0) {
    typedef std::vector<LinkBoardSpec>::const_iterator ILB;
    for (const auto& theLB : theLBs)
      str << theLB.print(depth);
  }
  return str.str();
}

void LinkConnSpec::add(const LinkBoardSpec& lb) { theLBs.push_back(lb); }

const LinkBoardSpec* LinkConnSpec::linkBoard(int linkBoardNumInLink) const {
  //FIXME - temporary implementaion, to be replace by LUT (in preparation)
  typedef std::vector<LinkBoardSpec>::const_iterator IT;
  for (const auto& theLB : theLBs) {
    if (linkBoardNumInLink == theLB.linkBoardNumInLink())
      return &theLB;
  }
  return nullptr;
}
