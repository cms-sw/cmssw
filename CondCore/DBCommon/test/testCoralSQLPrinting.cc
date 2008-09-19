//coral includes
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IMonitoringService.h"
#include "RelationalAccess/IMonitoringReporter.h"
#include "CoralBase/AttributeSpecification.h"
#include "SealKernel/Context.h"
#include "SealKernel/ComponentLoader.h"
#include "SealKernel/IMessageService.h"
#include "SealKernel/PropertyManager.h"
#include "PluginManager/PluginManager.h"
#include <iostream>

int main() {

  seal::Context* ctx( new seal::Context );

  seal::PluginManager* pm = seal::PluginManager::get();
  pm->initialise();
  seal::Handle<seal::ComponentLoader> loader = new seal::ComponentLoader( ctx );

  seal::IHandle<seal::IMessageService> msg = ctx->query<seal::IMessageService>( "SEAL/Services/MessageService" );
  if( !msg ){
    loader->load( "SEAL/Services/MessageService" );
    std::vector< seal::Handle<seal::IMessageService> > v_msgSvc;
    ctx->query( v_msgSvc );
    if ( ! v_msgSvc.empty() ) {
      seal::Handle<seal::IMessageService>& msgSvc = v_msgSvc.front();
      msgSvc->setOutputLevel( seal::Msg::Debug);
    }
  }
  
  loader->load( "CORAL/Services/ConnectionService"  );
  //loader->load( "CORAL/Services/MonitoringService");
  loader->load( "COND/Services/SQLMonitoringService");
  loader->load( "COND/Services/XMLAuthenticationService");
  size_t nchildren=loader->context()->children();
  for( size_t i=0; i<nchildren; ++i ){
    seal::Handle<seal::PropertyManager> pmgr=loader->context()->child(i)->component<seal::PropertyManager>();
    std::string scopeName=pmgr->scopeName();
    if( scopeName=="COND/Services/XMLAuthenticationService" ){
      //pmgr->property("AuthenticationFile")->set(std::string("authentication.xml"));
      pmgr->property("AuthenticationFile")->set(std::string("/build/gg/key.dat"));
    }
  }
  
  std::vector< seal::IHandle<coral::IConnectionService> > v_svc;
  ctx->query( v_svc );
  coral::IConnectionService* connectionService = v_svc.front().get();


  connectionService->configuration().setMonitoringLevel(coral::monitor::Trace);
  
  //std::string connectionString("sqlite_file:mytest.db");
  std::string connectionString("oracle://cms_orcoff_int2r/CMS_COND_PRESH");
  
  coral::ISessionProxy* session = connectionService->connect( connectionString );
  session->transaction().start();
  // creates a dummy table to be looked up
  std::string T1("TEST_DROP_ME");
  session->nominalSchema().dropIfExistsTable( T1 );
  coral::TableDescription descr;
  descr.setName( T1 );
  descr.insertColumn("N_X",coral::AttributeSpecification::typeNameForType<int>());
  descr.insertColumn("N_S",coral::AttributeSpecification::typeNameForType<std::string>());
  session->nominalSchema().createTable( descr );
  session->nominalSchema().dropIfExistsTable( T1 );
  session->transaction().commit();
  delete session;
  //connectionService->monitoringReporter().report();
  //std::cout << "Available reporter : " << std::endl;
  //std::set< std::string > rep = connectionService->monitoringReporter().monitoredDataSources();
  //std::set< std::string >::iterator iter;  
  //for ( iter = rep.begin( ); iter != rep.end( ); iter++ )
  //    std::cout << "reporter : " << *iter << std::endl;
  std::cout << "SQL Monitoring report for session" << std::endl;
  connectionService->monitoringReporter().reportToOutputStream( connectionString, std::cout );

  
  return 0;

}
