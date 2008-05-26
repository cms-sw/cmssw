#ifndef CondCore_Modules_EmptyIOVSource_h
#define CondCore_Modules_EmptyIOVSource_h

#include "FWCore/Framework/interface/ConfigurableInputSource.h"
#include "CondCore/DBCommon/interface/Time.h"
#include <set>
namespace cond {
  class EmptyIOVSource : public edm::ConfigurableInputSource {
  public:
    EmptyIOVSource(edm::ParameterSet const&, edm::InputSourceDescription const&);
    ~EmptyIOVSource();
  private:
    virtual bool produce(edm::Event & e);
    virtual void setRunAndEventInfo();
  private:
    std::string m_timeType;
    unsigned long long m_firstValid;
    unsigned long long m_lastValid;
    //unsigned long long m_currentValid;
    unsigned long long m_interval;
    std::set<cond::Time_t> m_iovs;
    std::set<cond::Time_t>::iterator m_iovit;
    std::set<cond::Time_t>::iterator m_current;
  };
}
#endif
