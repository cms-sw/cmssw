#ifndef INCLUDE_ORA_MONITORING_H
#define INCLUDE_ORA_MONITORING_H

//
#include <string>
#include <map>
#include "boost/date_time/posix_time/posix_time.hpp"
#include <iostream>

namespace ora {

  struct TransactionMonitoringData {
    explicit TransactionMonitoringData( boost::posix_time::ptime start );
    void stop(bool commit_status=true);
    boost::posix_time::ptime m_start;
    boost::posix_time::ptime m_stop;
    bool m_status;
  };

  class SessionMonitoringData {
  public:
    SessionMonitoringData( boost::posix_time::ptime start, const std::string& connectionString );
    ~SessionMonitoringData();
    size_t newTransaction();
    size_t stopTransaction( bool commit_status=true);
    void stop();
    size_t numberOfTransactions() const ;
    void report( std::ostream& out ) const ;
    std::string m_connectionString;
    boost::posix_time::ptime m_start;
    boost::posix_time::ptime m_stop;
  private:
    TransactionMonitoringData* m_currentTransaction;
    std::map<std::string,TransactionMonitoringData*> m_transactions;
    size_t m_ncur;
  };

  class Monitoring {
  public:
    static Monitoring& get(); 
    static bool isEnabled();
    static void enable();
    static std::string& outFileName();
  public:

    /// 
    virtual ~Monitoring() throw();

    SessionMonitoringData* startSession( const std::string& connectionString );

    void report( std::ostream& out );
  private:
    static bool s_enabled;
 
  private:
    // 
    Monitoring();
  private:

    std::map<std::string,SessionMonitoringData*> m_sessions;
  };

}

#endif
