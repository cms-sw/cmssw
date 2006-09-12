#ifndef CondCore_Modules_IOVSource_h
#define CondCore_Modules_IOVSource_h

#include "FWCore/Framework/interface/ConfigurableInputSource.h"
#include <set>
//#include "DataFormats/Common/interface/Timestamp.h"

namespace cond {
  class IOVSource : public edm::ConfigurableInputSource {
  public:
    IOVSource(edm::ParameterSet const&, edm::InputSourceDescription const&);
    ~IOVSource();
  private:
    virtual bool produce(edm::Event & e);
    virtual void setRunAndEventInfo();
  private:
    std::string m_connect;
    std::string m_catconnect;
    std::string m_timeType;
    unsigned int m_firstValid;
    unsigned int m_lastValid;
    unsigned int m_currentValid;
    std::set<unsigned long long> m_iovs;
    std::set<unsigned long long>::iterator m_iovit;
    std::set<unsigned long long>::iterator m_iovstop;
  };
}
#endif
