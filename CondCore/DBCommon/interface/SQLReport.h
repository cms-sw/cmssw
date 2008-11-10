#ifndef CondCoreDBCommon_SQLReport_H
#define CondCoreDBCommon_SQLReport_H
#include <string>
#include <sstream>

namespace cond {

  class DBSession;
  
  class SQLReport {

    public:

    explicit SQLReport(DBSession& session);

    virtual ~SQLReport(){}
    
    void reportForConnection(const std::string& connectionString);

    bool putOnFile(std::string fileName=std::string(""));

    private:

    SQLReport();

    DBSession& m_session;
    
    std::stringstream m_report;
    
  };
}

inline
cond::SQLReport::SQLReport(DBSession& session):m_session(session),m_report(){
}

#endif //  CondCoreDBCommon_SQLReport_H

  
