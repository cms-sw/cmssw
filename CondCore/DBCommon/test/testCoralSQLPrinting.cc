//local includes
#include "CondCore/DBCommon/interface/CoralServiceManager.h"

//CMSSW includes
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

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
#include <cstdlib>
#include <string>

int main() {
  std::string monitoringServiceName = "COND/Services/SQLMonitoringService";
  std::string authServiceName = "COND/Services/XMLAuthenticationService";
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::CoralServiceManager m_pluginManager;
  std::string authpath("/afs/cern.ch/cms/DB/conddb");
  std::string pathenv(std::string("CORAL_AUTH_PATH=")+authpath);
  ::putenv(const_cast<char*>(pathenv.c_str()));
  coral::MessageStream::setMsgVerbosity( coral::Debug );
  coral::Context::instance().loadComponent("CORAL/Services/ConnectionService");
  coral::Context::instance().loadComponent(monitoringServiceName, &m_pluginManager);
  coral::Context::instance().loadComponent(authServiceName, &m_pluginManager);
  //coral::Context::instance().PropertyManager().property("AuthenticationFile")->set(std::string("/build/gg/key.dat"));
  coral::IHandle<coral::IConnectionService> connectionService=coral::Context::instance().query<coral::IConnectionService>();
  connectionService->configuration().setAuthenticationService( authServiceName );
  connectionService->configuration().setMonitoringService( monitoringServiceName );
  connectionService->configuration().setMonitoringLevel(coral::monitor::Trace);	  
  std::string connectionString("oracle://cms_orcoff_prep/CMS_COND_UNIT_TESTS");
  
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
  connectionService->monitoringReporter().report();
  std::cout << "Available reporter : " << std::endl;
  std::set< std::string > rep = connectionService->monitoringReporter().monitoredDataSources();
  std::set< std::string >::iterator iter;  
  for ( iter = rep.begin( ); iter != rep.end( ); iter++ )
      std::cout << "reporter : " << *iter << std::endl;
  std::cout << "SQL Monitoring report for session" << std::endl;
  connectionService->monitoringReporter().reportToOutputStream( connectionString, std::cout );

  
  return 0;

}
