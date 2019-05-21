#ifndef L1TriggerConfig_L1TConfigProducers_QueryHelper_h
#define L1TriggerConfig_L1TConfigProducers_QueryHelper_h

#include <map>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include "CondTools/L1Trigger/interface/OMDSReader.h"

namespace l1t {

  // The following class encloses some of the conventions for the online DB model:
  //  https://indico.cern.ch/event/591003/contributions/2384788/attachments/1378957/2095301/L1TriggerDatabase_v2.pdf

  class OnlineDBqueryHelper {
  public:
    static std::map<std::string, std::string> fetch(const std::vector<std::string> &queryColumns,
                                                    const std::string &table,
                                                    const std::string &key,
                                                    l1t::OMDSReader &m_omdsReader);
  };

}  // namespace l1t
#endif
