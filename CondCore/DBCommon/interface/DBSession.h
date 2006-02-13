#ifndef COND_DBSESSION_H
#define COND_DBSESSION_H
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include <string>
namespace pool{
  class IDataSvc;
  class IFileCatalog;
}
namespace cond{
  class DBSession{
  public:
    explicit DBSession( const std::string& con );
    DBSession( const std::string& con, 
	       const std::string& catalogcon
	       );
    ~DBSession();
    void setCatalog( const std::string& catalogcon );
    void connect(  cond::ConnectMode mode=cond::ReadWriteCreate );
    void disconnect();
    void startUpdate();
    void startReadOnly();
    void commit();
    void rollback();
    const std::string connectionString() const;
    pool::IDataSvc& DataSvc() const;
    pool::IFileCatalog& Catalog() const;
  private:
    std::string m_con;
    pool::IFileCatalog* m_cat;
    pool::IDataSvc* m_svc;
    std::string m_catalogcon;
  };
}//ns cond
#endif
// DBSESSION_H
