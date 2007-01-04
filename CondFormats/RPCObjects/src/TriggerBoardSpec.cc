#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include <sstream>

TriggerBoardSpec::TriggerBoardSpec(int num) : theNum(num)
{ }

const LinkConnSpec * TriggerBoardSpec::linkConn(int tbInputNumber) const
{
  //FIXME - temporary implementaion, to be replace by LUT (in preparation)
  typedef std::vector<LinkConnSpec>::const_iterator IT;
  for (IT it=theLinks.begin(); it != theLinks.end(); it++) {
    if(tbInputNumber==it->triggerBoardInputNumber()) return &(*it);
  }
  return 0;
}

std::string TriggerBoardSpec::print(int depth) const
{
  std::ostringstream str;
  str << "TriggerBoardSpec: num=" << dccInputChannelNum() << std::endl;
  typedef std::vector<LinkConnSpec>::const_iterator ICON;
  depth--;
  if (depth >= 0) {
    for (ICON ic = theLinks.begin(); ic != theLinks.end(); ic++) str << (*ic).print(depth);
  }
  return str.str();
}

void TriggerBoardSpec::add(const LinkConnSpec & lc)
{
  theLinks.push_back(lc);
}
