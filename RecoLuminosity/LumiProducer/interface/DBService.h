#ifndef RecoLuminosity_LumiProducer_DBService_h
#define RecoLuminosity_LumiProducer_DBService_h
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
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
      DBService(const edm::ParameterSet& iConfig, 
		edm::ActivityRegistry& iAR);
      ~DBService();
      void postEndJob();
      void preEventProcessing( const edm::EventID & evtID, 
      			       const edm::Timestamp & iTime );
      void preModule(const edm::ModuleDescription& desc);
      void postModule(const edm::ModuleDescription& desc);
      void preBeginLumi(const edm::LuminosityBlockID&, 
			const edm::Timestamp& );
      coral::ISessionProxy* connectReadOnly( const std::string& connectstring );
      void disconnect( coral::ISessionProxy* session );
      lumi::DBConfig&  DBConfig();
      void setupWebCache();
    private:
      coral::ConnectionService* m_svc;
      lumi::DBConfig* m_dbconfig;
    };//cl DBService
  }//ns service
}//ns lumi
#endif
