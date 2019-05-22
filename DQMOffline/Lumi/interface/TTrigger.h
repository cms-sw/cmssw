#ifndef DQMOFFLINE_LUMI_TTRIGGER_H
#define DQMOFFLINE_LUMI_TTRIGGER_H

#include "DQMOffline/Lumi/interface/TriggerRecord.h"       // class to handle user specified trigger info
#include "DQMOffline/Lumi/interface/TriggerDefs.h"

namespace ZCountingTrigger
{
  class TTrigger 
  {
    public:
      TTrigger(const std::vector<std::string> &muonTriggerNames, const std::vector<std::string> &muonTriggerObjectNames);
      ~TTrigger(){}

      // Methods
      int  getTriggerBit(const std::string &iName) const;
      int  getTriggerObjectBit(const std::string &iName, const std::string &iObjName) const;
      bool pass(const std::string &iName, const TriggerBits &iTrig) const;
      bool passObj(const std::string &iName, const std::string &iObjName, const TriggerObjects &iTrigObj) const;

      std::vector<ZCountingTrigger::TriggerRecord> fRecords;
  };
}
#endif
