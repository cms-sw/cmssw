#ifndef DQMOFFLINE_LUMI_TRIGGERRECORD_H
#define DQMOFFLINE_LUMI_TRIGGERRECORD_H

#include <vector>
#include <string>
#include <utility>

namespace ZCountingTrigger {

class TriggerRecord
{
public:
  TriggerRecord(const std::string &name="", const unsigned int value=0) {
    hltPattern   = name;
    baconTrigBit = value;
    hltPathName  = "";
    hltPathIndex = (unsigned int)-1;
  }
  ~TriggerRecord(){}

  std::string	hltPattern;    // HLT path name/pattern (wildcards allowed: *,?)
  unsigned int  baconTrigBit;  // bacon trigger bit
  std::string	hltPathName;   // HLT path name in trigger menu
  unsigned int  hltPathIndex;  // HLT path index in trigger menu

  // map between trigger object name and bacon trigger object bit
  std::vector< std::pair<std::string,unsigned int> > objectMap;
};

}
#endif
