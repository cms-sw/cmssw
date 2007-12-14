#ifndef COND_DBCommon_DBSession_h
#define COND_DBCommon_DBSession_h
namespace cond{
  class ServiceLoader;
  class ConnectionConfiguration;
  class SessionConfiguration;
  /*
  **/
  class DBSession{
  public:
    DBSession();
    explicit DBSession( bool usePoolContext );
    ~DBSession();
    void open();
    void close();
    ServiceLoader& serviceLoader();
    ConnectionConfiguration& connectionConfiguration();
    SessionConfiguration& sessionConfiguration();
    bool isActive() const;
    void purgeConnectionPool();
  private:
    bool m_isActive;
    ServiceLoader* m_loader;
    ConnectionConfiguration* m_connectConfig;
    SessionConfiguration* m_sessionConfig;
    bool m_usePoolContext;
  };
}//ns cond
#endif
// DBSESSION_H
