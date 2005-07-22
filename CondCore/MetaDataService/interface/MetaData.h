#ifndef COND_METADATA_H
#define COND_METADATA_H
#include <string>
#include <memory>
#include "RelationalAccess/IRelationalSession.h"
#include "SealKernel/MessageStream.h"
namespace pool{
  class IRelationalTable;
}
namespace cond{
  class MetaData {
  public:
    MetaData(const std::string& contact);
    ~MetaData();
    bool addMapping(const std::string& name, const std::string& token);
    const std::string getToken( const std::string& name );
  private:
    void createTable(const std::string& tabname);
    std::string m_con;
    std::auto_ptr< pool::IRelationalSession > m_session;
    std::auto_ptr< seal::MessageStream > m_log;
    pool::IRelationalTable* m_table;
  };
}
#endif
