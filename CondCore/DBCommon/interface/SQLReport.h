#ifndef CondCoreDBCommon_SQLReport_H
#define CondCoreDBCommon_SQLReport_H
#include <string>
#include <sstream>

namespace cond {

  class DBSession;
  
  class SQLReport {

    public:

    explicit SQLReport(const DBSession& session);

    virtual ~SQLReport(){}
    
    void reportForConnection(const std::string& connectionString);

    bool putOnFile(std::string fileName=std::string(""));

    private:

    SQLReport();

    const DBSession& m_session;
    
    std::stringstream m_report;
    
  };
}

inline
cond::SQLReport::SQLReport(const DBSession& session):m_session(session),m_report(){
}

#endif //  CondCoreDBCommon_SQLReport_H

  
