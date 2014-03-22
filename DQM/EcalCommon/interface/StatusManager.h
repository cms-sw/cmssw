#ifndef StatusManager_H
#define StatusManager_H

#include <iostream>
#include <cstdint>
#include <vector>
#include <map>

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"

namespace ecaldqm
{

  class StatusManager {
  public:
    StatusManager();
    ~StatusManager() {}

    void readFromStream(std::istream&);
    void readFromObj(EcalDQMChannelStatus const&, EcalDQMTowerStatus const&);
    void writeToStream(std::ostream&) const;
    void writeToObj(EcalDQMChannelStatus&, EcalDQMTowerStatus&) const;

    uint32_t getStatus(uint32_t) const;

  private:
    std::map<std::string, uint32_t> dictionary_;
    std::map<uint32_t, uint32_t> status_;
  };

}

#endif
