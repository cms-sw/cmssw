#ifndef Container_h
#define Container_h

/*
 *      file:           Container.h
 *      Author:         Viktor Khristenko
 *
 *      Description:
 *              Container Base class
 *
 *
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"
#include "DQM/HcalCommon/interface/Logger.h"

#include <string>
#include <vector>

namespace hcaldqm {
  class Container {
  public:
    typedef dqm::legacy::DQMStore DQMStore;
    typedef dqm::legacy::MonitorElement MonitorElement;

    Container() : _folder("HcalInfo"), _qname("SomeQuantity") {}
    Container(std::string const &folder, std::string const &qname) : _folder(folder), _qname(qname) {}
    virtual ~Container() {}

    virtual void initialize(std::string const &folder, std::string const &qname, int debug = 0) {
      _folder = folder;
      _qname = qname;
      _logger.set(_qname, debug);
    }

  protected:
    std::string _folder;
    std::string _qname;
    Logger _logger;
  };
}  // namespace hcaldqm

#endif
