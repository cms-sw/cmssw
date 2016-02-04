#ifndef CondCoreDBCommon_SQLReport_H
#define CondCoreDBCommon_SQLReport_H
#include <string>
#include <sstream>
#include "CondCore/DBCommon/interface/DbConnection.h"

namespace cond {

  class DbConnection;
  
  class SQLReport {

    public:

    explicit SQLReport(DbConnection& connection);

    virtual ~SQLReport(){}
    
    void reportForConnection(const std::string& connectionString);

    bool putOnFile(std::string fileName=std::string(""));

    private:

    SQLReport();

    DbConnection m_connection;
    
    std::stringstream m_report;
    
  };
}

inline
cond::SQLReport::SQLReport(DbConnection& connection):m_connection(connection),m_report(){
}

#endif //  CondCoreDBCommon_SQLReport_H

  
