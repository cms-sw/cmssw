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
#include "CoralKernel/Context.h"
#include "CoralKernel/IHandle.h"
#include "CoralKernel/IProperty.h"
#include "CoralKernel/IPropertyManager.h"
#include "CoralBase/MessageStream.h"
#include <iostream>

int main() {

  coral::MessageStream::setMsgVerbosity( coral::Debug );
  coral::Context::instance().loadComponent("CORAL/Services/ConnectionService");
  coral::Context::instance().loadComponent("COND/Services/SQLMonitoringService");
  coral::Context::instance().loadComponent("COND/Services/XMLAuthenticationService");
  //coral::Context::instance().PropertyManager().property("AuthenticationFile")->set(std::string("/build/gg/key.dat"));
  coral::IHandle<coral::IConnectionService> connectionService=coral::Context::instance().query<coral::IConnectionService>();
  connectionService->configuration().setMonitoringLevel(coral::monitor::Trace);	  //std::string connectionString("sqlite_file:mytest.db");
  std::string connectionString("oracle://cms_orcoff_prep/CMS_COND_PRESH");
  
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
