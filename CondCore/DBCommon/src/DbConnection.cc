//local includes
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
// coral includes
#include "CoralKernel/Context.h"
#include "CoralKernel/IHandle.h"
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"

namespace cond {
  class DbConnection::ConnectionImpl {
    public:
      ConnectionImpl();

      virtual ~ConnectionImpl();

      void close();

      bool m_initialized;
      coral::ConnectionService* m_connectionService;
      DbConnectionConfiguration m_configuration;
  };  
}

cond::DbConnection::ConnectionImpl::ConnectionImpl():
  m_initialized(false),m_connectionService(0),m_configuration(){ 
  m_connectionService = new coral::ConnectionService;
}

cond::DbConnection::ConnectionImpl::~ConnectionImpl(){
  close();
}


void cond::DbConnection::ConnectionImpl::close()
{
  if(m_connectionService) {
    delete m_connectionService;
    m_connectionService = 0;
  }
}

cond::DbConnection::DbConnection():
  m_implementation(new ConnectionImpl()){
  configure();
}

cond::DbConnection::DbConnection(const DbConnection& conn):
  m_implementation( conn.m_implementation ){
}

cond::DbConnection::~DbConnection(){
}

cond::DbConnection& cond::DbConnection::operator=(const cond::DbConnection& conn)
{
  if(this!=&conn) m_implementation = conn.m_implementation;
  return *this;
}

void cond::DbConnection::configure()
{
  m_implementation->m_configuration.configure( m_implementation->m_connectionService->configuration() );
  m_implementation->m_initialized = true;
}

void cond::DbConnection::configure( cond::DbConfigurationDefaults defaultItem ){
  m_implementation->m_configuration = DbConnectionConfiguration::defaultConfigurations()[ defaultItem ];
  configure();  
}

void 
cond::DbConnection::configure( const edm::ParameterSet& connectionPset ){
  m_implementation->m_configuration.setParameters( connectionPset );
  configure();
}

cond::DbSession cond::DbConnection::createSession() const{
  if(!isOpen())
    throw cond::Exception("DbConnection::createSession: cannot create session. Connection has been closed.");
  return cond::DbSession( *this );
}


void cond::DbConnection::close()
{
  m_implementation->close();
}

bool cond::DbConnection::isOpen() const
{
  return m_implementation->m_connectionService;
}
  
cond::DbConnectionConfiguration & cond::DbConnection::configuration()
{
  return m_implementation->m_configuration;
}

cond::DbConnectionConfiguration const & cond::DbConnection::configuration() const
{
  return m_implementation->m_configuration;
}

coral::IConnectionService& cond::DbConnection::connectionService() const {
  if(!isOpen())
    throw cond::Exception("DbConnection::connectionService: cannot get connection service. Connection has not been open.");
  return *m_implementation->m_connectionService;
}

const coral::IMonitoringReporter& cond::DbConnection::monitoringReporter() const {
  if(!isOpen())
    throw cond::Exception("DbConnection::monitoringReporter: cannot get connection service. Connection has not been open.");
  return m_implementation->m_connectionService->monitoringReporter();
}

coral::IWebCacheControl& cond::DbConnection::webCacheControl() const{
  if(!isOpen())
    throw cond::Exception("DbConnection::webCacheControl: cannot get connection service. Connection has not been open.");
  return m_implementation->m_connectionService->webCacheControl();
}

