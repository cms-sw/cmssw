#include "DQM/Integration/interface/TestBase.h"
#include "RelationalAccess/IRelationalDomain.h"
#include "RelationalAccess/IConnection.h"
#include "RelationalAccess/ISession.h"
#include "RelationalAccess/RelationalServiceException.h"
//#include "PluginManager/PluginManager.h"
#include "CoralKernel/Context.h"
//#include "SealKernel/ComponentLoader.h"

TestBase::TestBase():
  //  m_context( new seal::Context ),
  m_connection( 0 )
{
  //seal::PluginManager* pm = seal::PluginManager::get();
  //pm->initialise();
  //seal::Handle<seal::ComponentLoader> loader = new seal::ComponentLoader( m_context.get() );
  //if ( ! loader->load( "CORAL/RelationalPlugins/oracle" ) ) {
  //  throw std::runtime_error( "Could not load the OracleAccess plugin" );
  // }
}


TestBase::~TestBase(){
  if ( m_connection ) delete m_connection;
  //m_context = 0;
}


coral::ISession*
TestBase::connect( const std::string& connectionString,
                   const std::string& user,
                   const std::string& pass )
{
  // seal::IHandle<coral::IRelationalDomain> iHandle = m_context->query<coral::IRelationalDomain>( "CORAL/RelationalPlugins/oracle" );
  //if ( ! iHandle ) {
  // throw coral::NonExistingDomainException( "oracle" );
 
  coral::Context& ctx = coral::Context::instance();
  coral::IHandle<coral::IRelationalDomain> iHandle=ctx.query<coral::IRelationalDomain>("CORAL/RelationalPlugins/oracle");
      if ( ! iHandle.isValid() ) {
      throw std::runtime_error( "Could not load the OracleAccess plugin" );
    }

 

  std::pair<std::string, std::string> connectionAndSchema = iHandle->decodeUserConnectionString( connectionString );

  if ( ! m_connection )
    m_connection = iHandle->newConnection( connectionAndSchema.first );

  if ( ! m_connection->isConnected() )
    m_connection->connect();

  coral::ISession* session = m_connection->newSession( connectionAndSchema.second );

  if ( session ) {
    session->startUserSession( user, pass );
  }

  return session;
}


void
TestBase::setVerbosityLevel( coral::MsgLevel level )
{
  coral::MessageStream::setMsgVerbosity(level);
}
    

