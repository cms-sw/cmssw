#ifndef Mapper_h
#define Mapper_h

/*
 *	file:		Mapper.h
 *	Author:		Viktor Khristenko
 *
 *	Description:
 */

#include "DQM/HcalCommon/interface/Constants.h"
#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"
#include "DQM/HcalCommon/interface/Logger.h"

#include <sstream>
#include <string>
#include <vector>

namespace hcaldqm {
  namespace mapper {
    class Mapper {
    public:
      Mapper() {}
      virtual ~Mapper() {}

      virtual uint32_t getHash(HcalDetId const &) const { return 0; }
      virtual uint32_t getHash(HcalElectronicsId const &) const { return 0; }
      virtual uint32_t getHash(HcalTrigTowerDetId const &) const { return 0; }
      virtual uint32_t getHash(HcalTrigTowerDetId const &, HcalElectronicsId const &) const { return 0; }

      virtual std::string getName(HcalDetId const &) const { return ""; }
      virtual std::string getName(HcalElectronicsId const &) const { return ""; }
      virtual std::string getName(HcalTrigTowerDetId const &) const { return ""; }
      virtual std::string getName(HcalTrigTowerDetId const &, HcalElectronicsId const &) const { return ""; }

    protected:
    };
  }  // namespace mapper
}  // namespace hcaldqm

#endif
