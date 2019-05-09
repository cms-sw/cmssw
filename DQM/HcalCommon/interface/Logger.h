#ifndef Logger_h
#define Logger_h

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"

class Logger {
public:
  Logger(std::string const &name, int debug = 0) : _name(name), _debug(debug) {}
  Logger() {}
  virtual ~Logger() {}

  inline void dqmthrow(std::string const &msg) const { throw cms::Exception("HCALDQM") << _name << "::" << msg; }
  inline void warn(std::string const &msg) const { edm::LogWarning("HCALDQM") << _name << "::" << msg; }
  inline void info(std::string const &msg) const {
    if (_debug == 0)
      return;
    edm::LogInfo("HCALDQM") << _name << "::" << msg;
  }
  template <typename STDTYPE>
  inline void debug(STDTYPE const &msg) const {
    if (_debug == 0)
      return;

    std::cout << "%MSG" << std::endl;
    std::cout << "$MSG-d HCALDQM::" << _name << "::" << msg;
    std::cout << std::endl;
  }

  inline void set(std::string const &name, int debug = 0) {
    _name = name;
    _debug = debug;

    if (debug == 0)
      return;

    this->debug("Setting up Logger for " + _name);
  }

protected:
  std::string _name;
  int _debug;
};

#endif
