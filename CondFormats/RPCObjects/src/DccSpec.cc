#include "CondFormats/RPCObjects/interface/DccSpec.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
  return (theId >=0) ? &theTBs[channelNumber-MIN_CHANNEL_NUMBER] : 0;
}

void DccSpec::add(const TriggerBoardSpec & tb) 
{
  if (theTBs.empty()) theTBs.resize(NUMBER_OF_CHANNELS);
  int channel = tb.dccInputChannelNum();
  if (    channel >= MIN_CHANNEL_NUMBER  
       && channel <= NUMBER_OF_CHANNELS+MIN_CHANNEL_NUMBER-1) {
    theTBs[channel-MIN_CHANNEL_NUMBER] = tb;
  } else {
     edm::LogInfo(" incorrect tb, skipp adding.")<<"\t id="<<channel; 
  }
}
