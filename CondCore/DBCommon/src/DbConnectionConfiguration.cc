//local includes
#include "CondCore/DBCommon/interface/DbConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/CoralServiceManager.h"
#include "CondCore/DBCommon/interface/Auth.h"
// CMSSW includes
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"
// coral includes
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "CoralKernel/Context.h"
#include "CoralKernel/IProperty.h"
#include "CoralKernel/IPropertyManager.h"
// externals
#include <boost/filesystem/operations.hpp>

std::vector<cond::DbConnectionConfiguration>&
cond::DbConnectionConfiguration::defaultConfigurations(){
  static std::vector<DbConnectionConfiguration> s_defaultConfigurations;
  // coral default
  s_defaultConfigurations.push_back( cond::DbConnectionConfiguration() );
  // cms default
  s_defaultConfigurations.push_back( cond::DbConnectionConfiguration( false, 0, false, 10, 60, false, "", "", coral::Error, coral::monitor::Off, false ) );
  // prod default
  s_defaultConfigurations.push_back( cond::DbConnectionConfiguration( false, 0, false, 10, 60, false, "", "", coral::Error, coral::monitor::Off, false ) );
  // tool default
  s_defaultConfigurations.push_back( cond::DbConnectionConfiguration( false, 0, false, 10, 60, false, "", "",coral::Error, coral::monitor::Off, false ) );
  // web default
  s_defaultConfigurations.push_back( cond::DbConnectionConfiguration( false, 0, false, 10, 60, false, "", "",coral::Error, coral::monitor::Off, false ) );
  return s_defaultConfigurations;
}

cond::DbConnectionConfiguration::DbConnectionConfiguration():
  m_connectionSharing(false,false),
  m_connectionTimeOut(false,0),
  m_readOnlySessionOnUpdateConnections(false,false),
  m_connectionRetrialPeriod(false,0),
  m_connectionRetrialTimeOut(false,0),
  m_poolAutomaticCleanUp(false,false),
  m_authPath(""),
  m_authSys(0),
  m_transactionId(),
  m_messageLevel(coral::Error),
  m_monitoringLevel(coral::monitor::Off),
  m_SQLMonitoring(false),
  m_pluginManager(new CoralServiceManager){
}

cond::DbConnectionConfiguration::DbConnectionConfiguration( bool connectionSharing,
                                                            int connectionTimeOut,
                                                            bool readOnlySessionOnUpdateConnections,
                                                            int connectionRetrialPeriod,
                                                            int connectionRetrialTimeOut,
                                                            bool poolAutomaticCleanUp,
                                                            const::std::string& authenticationPath,
                                                            const::std::string& transactionId,
							    coral::MsgLevel msgLev,
                                                            coral::monitor::Level monitorLev,
                                                            bool SQLMonitoring ):
  m_connectionSharing(true,connectionSharing),
  m_connectionTimeOut(true,connectionTimeOut),
  m_readOnlySessionOnUpdateConnections(true,readOnlySessionOnUpdateConnections),
  m_connectionRetrialPeriod(true,connectionRetrialPeriod),
  m_connectionRetrialTimeOut(true,connectionRetrialTimeOut),
  m_poolAutomaticCleanUp(true,poolAutomaticCleanUp),
  m_authPath(authenticationPath),
  m_authSys(0),
  m_transactionId(transactionId),
  m_messageLevel(msgLev),
  m_monitoringLevel(monitorLev),
  m_SQLMonitoring(SQLMonitoring),
  m_pluginManager(new CoralServiceManager){
}

cond::DbConnectionConfiguration::DbConnectionConfiguration( const cond::DbConnectionConfiguration& rhs):
  m_connectionSharing(rhs.m_connectionSharing),
  m_connectionTimeOut(rhs.m_connectionTimeOut),
  m_readOnlySessionOnUpdateConnections(rhs.m_readOnlySessionOnUpdateConnections),
  m_connectionRetrialPeriod(rhs.m_connectionRetrialPeriod),
  m_connectionRetrialTimeOut(rhs.m_connectionRetrialTimeOut),
  m_poolAutomaticCleanUp(rhs.m_poolAutomaticCleanUp),
  m_authPath(rhs.m_authPath),
  m_authSys(rhs.m_authSys),
  m_transactionId(rhs.m_transactionId),
  m_messageLevel(rhs.m_messageLevel),
  m_monitoringLevel(rhs.m_monitoringLevel),
  m_SQLMonitoring(rhs.m_SQLMonitoring),
  m_pluginManager(new CoralServiceManager){
}

cond::DbConnectionConfiguration::~DbConnectionConfiguration(){
  delete m_pluginManager;
}

cond::DbConnectionConfiguration&
cond::DbConnectionConfiguration::operator=( const cond::DbConnectionConfiguration& rhs){
  m_connectionSharing = rhs.m_connectionSharing;
  m_connectionTimeOut = rhs.m_connectionTimeOut;
  m_readOnlySessionOnUpdateConnections = rhs.m_readOnlySessionOnUpdateConnections;
  m_connectionRetrialPeriod = rhs.m_connectionRetrialPeriod;
  m_connectionRetrialTimeOut = rhs.m_connectionRetrialTimeOut;
  m_poolAutomaticCleanUp = rhs.m_poolAutomaticCleanUp;
  m_authPath = rhs.m_authPath;
  m_authSys = rhs.m_authSys;
  m_transactionId=rhs.m_transactionId;
  m_messageLevel = rhs.m_messageLevel;
  m_monitoringLevel = rhs.m_monitoringLevel;
  m_SQLMonitoring = rhs.m_SQLMonitoring;
  return *this;
}

void cond::DbConnectionConfiguration::setParameters( const edm::ParameterSet& connectionPset ){
  std::string authPath = connectionPset.getUntrackedParameter<std::string>("authenticationPath","");
  setAuthenticationPath(authPath);
  int authSysPar = connectionPset.getUntrackedParameter<int>("authenticationSystem",0);
  setAuthenticationSystem( authSysPar );
  setTransactionId(connectionPset.getUntrackedParameter<std::string>("transactionId",""));
  int messageLevel = connectionPset.getUntrackedParameter<int>("messageLevel",0);
  coral::MsgLevel level = coral::Error;
  switch (messageLevel) {
    case 0 :
      level = coral::Error;
      break;    
    case 1:
      level = coral::Warning;
      break;
    case 2:
      level = coral::Info;
      break;
    case 3:
      level = coral::Debug;
      break;
    default:
      level = coral::Error;
  }
  setMessageLevel(level);
  bool enableConnectionSharing = connectionPset.getUntrackedParameter<bool>("enableConnectionSharing",true);
  setConnectionSharing( enableConnectionSharing );
  int connectionTimeOut = connectionPset.getUntrackedParameter<int>("connectionTimeOut",600);
  setConnectionTimeOut( connectionTimeOut );
  bool enableReadOnlySessionOnUpdateConnection = connectionPset.getUntrackedParameter<bool>("enableReadOnlySessionOnUpdateConnection",true);
  setReadOnlySessionOnUpdateConnections( enableReadOnlySessionOnUpdateConnection );
  int connectionRetrialPeriod = connectionPset.getUntrackedParameter<int>("connectionRetrialPeriod",30);
  setConnectionRetrialPeriod( connectionRetrialPeriod );
  int connectionRetrialTimeOut = connectionPset.getUntrackedParameter<int>("connectionRetrialTimeOut",180);
  setConnectionRetrialTimeOut( connectionRetrialTimeOut );
  bool enablePoolAutomaticCleanUp = connectionPset.getUntrackedParameter<bool>("enablePoolAutomaticCleanUp",false);
  setPoolAutomaticCleanUp( enablePoolAutomaticCleanUp );
  //int idleConnectionCleanupPeriod = connectionPset.getUntrackedParameter<int>("idleConnectionCleanupPeriod",300);
}

void cond::DbConnectionConfiguration::setConnectionSharing( bool flag ){
  m_connectionSharing.first = true;
  m_connectionSharing.second = flag;
}

void cond::DbConnectionConfiguration::setConnectionTimeOut( int timeOut ){
  m_connectionTimeOut.first = true;
  m_connectionTimeOut.second = timeOut;
}

void cond::DbConnectionConfiguration::setReadOnlySessionOnUpdateConnections( bool flag ){
  m_readOnlySessionOnUpdateConnections.first = true;
  m_readOnlySessionOnUpdateConnections.second = flag;
}

void cond::DbConnectionConfiguration::setConnectionRetrialPeriod( int period ){
  m_connectionRetrialPeriod.first = true;
  m_connectionRetrialPeriod.second = period;
}

void cond::DbConnectionConfiguration::setConnectionRetrialTimeOut( int timeout ){
  m_connectionRetrialTimeOut.first = true;
  m_connectionRetrialTimeOut.second = timeout;
}

void cond::DbConnectionConfiguration::setPoolAutomaticCleanUp( bool flag ){
  m_poolAutomaticCleanUp.first = true;
  m_poolAutomaticCleanUp.second = flag;
}

void cond::DbConnectionConfiguration::setAuthenticationPath( const std::string& p ){
  m_authPath = p;
}

void cond::DbConnectionConfiguration::setAuthenticationSystem( int authSysCode ){
  m_authSys = authSysCode;
}

void cond::DbConnectionConfiguration::setTransactionId( std::string const & tid) {
  m_transactionId=tid;
}

void cond::DbConnectionConfiguration::setMessageLevel( coral::MsgLevel l ) {
  m_messageLevel = l; 
}

void cond::DbConnectionConfiguration::setMonitoringLevel( coral::monitor::Level l)
{
  m_monitoringLevel = l;  
}


void cond::DbConnectionConfiguration::setSQLMonitoring( bool flag ){
  m_SQLMonitoring = flag;
}

void cond::DbConnectionConfiguration::configure( coral::IConnectionServiceConfiguration& coralConfig) const 
{
  // message streaming
  coral::MessageStream::setMsgVerbosity( m_messageLevel );
  std::string authServiceName("CORAL/Services/EnvironmentAuthenticationService");
  std::string authPath = m_authPath;
  // authentication
  if( authPath.empty() ){
    // first try to check the env...
    const char* authEnv = ::getenv( Auth::COND_AUTH_PATH );
    if(authEnv){
      authPath += authEnv;
    } 
  }
  int authSys = m_authSys;
  // first attempt, look at the env...
  const char* authSysEnv = ::getenv( Auth::COND_AUTH_SYS );
  if( authSysEnv ){
    authSys = ::atoi( authSysEnv );
  }
  if( authSys != CondDbKey && authSys != CoralXMLFile ){
    // take the default
    authSys = CondDbKey;
  }  
  std::string servName("");
  if( authSys == CondDbKey ){
    if( authPath.empty() ){
      const char* authEnv = ::getenv("HOME");
      if(authEnv){
	authPath += authEnv;
      } 
    }
    servName = "COND/Services/RelationalAuthenticationService";     
    //edm::LogInfo("DbSessionInfo") << "Authentication using Keys";  
  } else if( authSys == CoralXMLFile ){
    if( authPath.empty() ){
      authPath = ".";
    }
    servName = "COND/Services/XMLAuthenticationService";  
    //edm::LogInfo("DbSessionInfo") << "Authentication using XML File";  
  }
  if( !authPath.empty() ){
    authServiceName = servName;    
    coral::Context::instance().PropertyManager().property(Auth::COND_AUTH_PATH_PROPERTY)->set(authPath);  
    coral::Context::instance().loadComponent( authServiceName, m_pluginManager );
  }
  coralConfig.setAuthenticationService( authServiceName );
  // connection sharing
  if(m_connectionSharing.first)
  {
    if(m_connectionSharing.second) coralConfig.enableConnectionSharing();
    else coralConfig.disableConnectionSharing();   
  }
  // connection timeout
  if(m_connectionTimeOut.first)
  {
    coralConfig.setConnectionTimeOut(m_connectionTimeOut.second);    
  }
  // read only session on update connection
  if(m_readOnlySessionOnUpdateConnections.first)
  {
    if(m_readOnlySessionOnUpdateConnections.second) coralConfig.enableReadOnlySessionOnUpdateConnections();
    else coralConfig.disableReadOnlySessionOnUpdateConnections();    
  }
  // pool automatic clean up
  if(m_poolAutomaticCleanUp.first)
  {
    if(m_poolAutomaticCleanUp.second) coralConfig.enablePoolAutomaticCleanUp();
    else coralConfig.disablePoolAutomaticCleanUp();
  }
  // connection retrial period
  if(m_connectionRetrialPeriod.first) 
  {
    coralConfig.setConnectionRetrialPeriod( m_connectionRetrialPeriod.second );    
  }
  // connection retrial timeout
  if( m_connectionRetrialTimeOut.first)
  {
    coralConfig.setConnectionRetrialTimeOut(m_connectionRetrialTimeOut.second );    
  }
  // monitoring level
  coralConfig.setMonitoringLevel( m_monitoringLevel );
  // SQL monitoring
  if( m_SQLMonitoring )
  {
    std::string pluginName("COND/Services/SQLMonitoringService");
    coral::Context::instance().loadComponent( pluginName, m_pluginManager );
    coralConfig.setMonitoringLevel(coral::monitor::Trace);   
  }
  
}

bool cond::DbConnectionConfiguration::isConnectionSharingEnabled() const 
{
  return m_connectionSharing.second;  
}

int cond::DbConnectionConfiguration::connectionTimeOut() const {
  return m_connectionTimeOut.second;  
}

bool cond::DbConnectionConfiguration::isReadOnlySessionOnUpdateConnectionEnabled() const {
  return m_readOnlySessionOnUpdateConnections.second;  
}

int cond::DbConnectionConfiguration::connectionRetrialPeriod() const {
  return m_connectionRetrialPeriod.second;  
}

int cond::DbConnectionConfiguration::connectionRetrialTimeOut() const {
  return m_connectionRetrialTimeOut.second;
}

bool cond::DbConnectionConfiguration::isPoolAutomaticCleanUpEnabled() const {
  return m_poolAutomaticCleanUp.second;  
}

const std::string& cond::DbConnectionConfiguration::authenticationPath() const 
{
  return m_authPath;
}

const std::string& cond::DbConnectionConfiguration::transactionId() const {
  return m_transactionId;
}


coral::MsgLevel cond::DbConnectionConfiguration::messageLevel() const 
{
  return m_messageLevel;
}

bool cond::DbConnectionConfiguration::isSQLMonitoringEnabled() const
{
  return m_SQLMonitoring;  
}

