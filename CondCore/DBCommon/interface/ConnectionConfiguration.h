#ifndef COND_DBCommon_ConnectionConfiguration_h
#define COND_DBCommon_ConnectionConfiguration_h
namespace cond{
  class ConnectionConfiguration{
  public:
    ConnectionConfiguration();
    ~ConnectionConfiguration();
    void enableConnectionSharing();
    bool isConnectionSharingEnabled() const;
    void setConnectionRetrialPeriod( int timeInSeconds );
    int connectionRetrialPeriod() const;
    void setConnectionRetrialTimeOut( int timeOutInSeconds );
    int connectionRetrialTimeOut() const;
    void setConnectionTimeOut( int timeOutInSeconds );
    int connectionTimeOut();
    void enableReadOnlySessionOnUpdateConnections();
    void disableReadOnlySessionOnUpdateConnections();
    bool isReadOnlySessionOnUpdateConnectionsEnabled();
  private:
    bool m_enableConSharing;
    int m_connectionRetrialPeriod;
    int m_connectionRetrialTimeOut;
    int m_connectionTimeOut;
    bool m_enableCommonConnection;
  };
}
#endif
