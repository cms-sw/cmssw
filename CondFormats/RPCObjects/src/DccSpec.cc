#include "CondFormats/RPCObjects/interface/DccSpec.h"
#include <iostream>

DccSpec::DccSpec(int id) : theId(id) 
{ }

void DccSpec::print(int depth) const
{
  std::cout << "DccSpec:id=" << id() << std::endl;
  typedef std::vector<TriggerBoardSpec>::const_iterator ITTB;
  depth--;
  for (ITTB it = theTBs.begin(); it != theTBs.end(); it++) it->print(depth);
}

const TriggerBoardSpec * DccSpec::triggerBoard(int channelNumber) const
{
  // FIXME - temporary implementaion, to be replace by LUT (in preparation)
  typedef std::vector<TriggerBoardSpec>::const_iterator ITTB;
  for (ITTB it = theTBs.begin(); it != theTBs.end(); it++) {
    if( channelNumber == it->dccInputChannelNum()) return &(*it);
  }
  return 0;
}

void DccSpec::add(const TriggerBoardSpec & tb) 
{
  theTBs.push_back(tb);
}
