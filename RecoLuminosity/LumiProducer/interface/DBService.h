#ifndef RecoLuminosity_LumiProducer_DBService_h
#define RecoLuminosity_LumiProducer_DBService_h
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
namespace coral{
  class ISessionProxy;
  class ConnectionService;
}
namespace lumi{
  class DBConfig;
  namespace service{
    class DBService{
    public:
      DBService(const edm::ParameterSet& iConfig);
      ~DBService();

      coral::ISessionProxy* connectReadOnly( const std::string& connectstring );
      void disconnect( coral::ISessionProxy* session );

    private:
      coral::ConnectionService* m_svc;
      lumi::DBConfig* m_dbconfig;
    };//cl DBService
  }//ns service
}//ns lumi
#endif
