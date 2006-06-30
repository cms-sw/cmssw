#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include <iostream>

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

void TriggerBoardSpec::print(int depth) const
{
  if (depth<0) return;
  std::cout << "TriggerBoardSpec: num=" << dccInputChannelNum() << std::endl;
  typedef std::vector<LinkConnSpec>::const_iterator ICON;
  depth--;
  for (ICON ic = theLinks.begin(); ic != theLinks.end(); ic++) 
    (*ic).print(depth);
}

void TriggerBoardSpec::add(const LinkConnSpec & lc)
{
  theLinks.push_back(lc);
}
