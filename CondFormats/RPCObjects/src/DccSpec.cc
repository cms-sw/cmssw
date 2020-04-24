#include "CondFormats/RPCObjects/interface/DccSpec.h"
#include <sstream>
#include <iostream>

DccSpec::DccSpec(int id) : theId(id) 
{ }

std::string DccSpec::print(int depth) const
{
  
  std::ostringstream str;
  str << "DccSpec:id=" << id() << std::endl;
  depth--;
  if (depth >= 0) {
    typedef std::vector<TriggerBoardSpec>::const_iterator ITTB;
    for (ITTB it = theTBs.begin(); it != theTBs.end(); it++) str << it->print(depth);
  }
  return str.str();
}

const TriggerBoardSpec * DccSpec::triggerBoard(int channelNumber) const
{
//  return (theId >=0) ? &theTBs[channelNumber-MIN_CHANNEL_NUMBER] : 0;

  //FIXME - temporary implementaion, to be replace by LUT (in preparation)
  typedef std::vector<TriggerBoardSpec>::const_iterator IT;
  for (IT it=theTBs.begin(); it != theTBs.end(); it++) {
    if(channelNumber ==it->dccInputChannelNum()) return &(*it);
  }
  return nullptr;

}

void DccSpec::add(const TriggerBoardSpec & tb) 
{
//  if (theTBs.empty()) theTBs.resize(NUMBER_OF_CHANNELS);
//  int channel = tb.dccInputChannelNum();
//  if (    channel >= MIN_CHANNEL_NUMBER  
//       && channel <= NUMBER_OF_CHANNELS+MIN_CHANNEL_NUMBER-1) {
//    theTBs[channel-MIN_CHANNEL_NUMBER] = tb;
//  } else {
//     edm::LogInfo(" incorrect tb, skipp adding.")<<"\t id="<<channel; 
//  }
  theTBs.push_back(tb);
}
