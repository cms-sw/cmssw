#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "DbConnectionString.h"
#include "IDbAuthentication.h"
#include "SessionImpl.h"
#include "IOVSchema.h"
#include "CoralMsgReporter.h"
//
#include "CondCore/CondDB/interface/CoralServiceManager.h"
#include "CondCore/CondDB/interface/Auth.h"
// CMSSW includes
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// coral includes
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/IAuthenticationService.h"
#include "RelationalAccess/IWebCacheControl.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "CoralKernel/Context.h"
#include "CoralKernel/IProperty.h"
#include "CoralKernel/IPropertyManager.h"

namespace cond {

  namespace persistency {

    ConnectionPool::ConnectionPool() {
      m_pluginManager = new cond::CoralServiceManager;
      m_msgReporter = new CoralMsgReporter;
      coral::MessageStream::installMsgReporter(m_msgReporter);
      configure();
    }

    ConnectionPool::~ConnectionPool() { delete m_pluginManager; }

    void ConnectionPool::setAuthenticationPath(const std::string& p) { m_authPath = p; }

    void ConnectionPool::setAuthenticationSystem(int authSysCode) { m_authSys = authSysCode; }

    void ConnectionPool::setFrontierSecurity(const std::string& signature) { m_frontierSecurity = signature; }

    void ConnectionPool::setLogging(bool flag) { m_loggingEnabled = flag; }

    void ConnectionPool::setParameters(const edm::ParameterSet& connectionPset) {
      //set the connection parameters from a ParameterSet
      //if a parameter is not defined, keep the values already set in the data members
      //(i.e. default if no other setters called, or the ones currently available)
      setAuthenticationPath(connectionPset.getUntrackedParameter<std::string>("authenticationPath", m_authPath));
      setAuthenticationSystem(connectionPset.getUntrackedParameter<int>("authenticationSystem", m_authSys));
      setFrontierSecurity(connectionPset.getUntrackedParameter<std::string>("security", m_frontierSecurity));
      int messageLevel =
          connectionPset.getUntrackedParameter<int>("messageLevel", 0);  //0 corresponds to Error level, current default
      coral::MsgLevel level;
      switch (messageLevel) {
        case 0:
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
      setMessageVerbosity(level);
      setConnectionTimeout(connectionPset.getUntrackedParameter<int>("connectionTimeout", m_connectionTimeout));
      setLogging(connectionPset.getUntrackedParameter<bool>("logging", m_loggingEnabled));
    }

    bool ConnectionPool::isLoggingEnabled() const { return m_loggingEnabled; }

    void ConnectionPool::configure(coral::IConnectionServiceConfiguration& coralConfig) {
      coralConfig.disablePoolAutomaticCleanUp();
      coralConfig.disableConnectionSharing();
      coralConfig.setConnectionTimeOut(m_connectionTimeout);
      // message streaming
      coral::MessageStream::setMsgVerbosity(m_messageLevel);
      m_msgReporter->setOutputLevel(m_messageLevel);

      // authentication
      m_authenticationService = std::string("CORAL/Services/EnvironmentAuthenticationService");
      std::string authPath = m_authPath;
      // authentication
      if (authPath.empty()) {
        // first try to check the env...
        const char* authEnv = std::getenv(cond::auth::COND_AUTH_PATH);
        if (authEnv) {
          authPath += authEnv;
        }
      }
      int authSys = m_authSys;
      // first attempt, look at the env...
      const char* authSysEnv = std::getenv(cond::auth::COND_AUTH_SYS);
      if (authSysEnv) {
        authSys = ::atoi(authSysEnv);
      }
      if (authSys != CondDbKey && authSys != CoralXMLFile) {
        // take the default
        authSys = CondDbKey;
      }
      std::string servName("");
      if (authSys == CondDbKey) {
        if (authPath.empty()) {
          const char* authEnv = std::getenv("HOME");
          if (authEnv) {
            authPath += authEnv;
          }
        }
        servName = "COND/Services/RelationalAuthenticationService";
      } else if (authSys == CoralXMLFile) {
        if (authPath.empty()) {
          authPath = ".";
        }
        servName = "COND/Services/XMLAuthenticationService";
      }
      if (!authPath.empty()) {
        m_authenticationService = servName;
        coral::Context::instance().PropertyManager().property(cond::auth::COND_AUTH_PATH_PROPERTY)->set(authPath);
        coral::Context::instance().loadComponent(m_authenticationService, m_pluginManager);
      }

      coralConfig.setAuthenticationService(m_authenticationService);
    }

    void ConnectionPool::configure() {
      coral::ConnectionService connServ;
      configure(connServ.configuration());
    }

    std::shared_ptr<coral::ISessionProxy> ConnectionPool::createCoralSession(const std::string& connectionString,
                                                                             const std::string& transactionId,
                                                                             bool writeCapable) {
      coral::ConnectionService connServ;
      //all sessions opened with this connection service will share the same frontier security option.
      std::pair<std::string, std::string> fullConnectionPars =
          getConnectionParams(connectionString, transactionId, m_frontierSecurity);
      if (!fullConnectionPars.second.empty()) {
        //all sessions opened with this connection service will share the same TTL settings for TAG, IOV, and PAYLOAD tables.
        connServ.webCacheControl().setTableTimeToLive(fullConnectionPars.second, TAG::tname, 1);
        connServ.webCacheControl().setTableTimeToLive(fullConnectionPars.second, IOV::tname, 1);
        connServ.webCacheControl().setTableTimeToLive(fullConnectionPars.second, PAYLOAD::tname, 3);
      }

      return std::shared_ptr<coral::ISessionProxy>(
          connServ.connect(fullConnectionPars.first,
                           writeCapable ? auth::COND_WRITER_ROLE : auth::COND_READER_ROLE,
                           writeCapable ? coral::Update : coral::ReadOnly));
    }

    Session ConnectionPool::createSession(const std::string& connectionString,
                                          const std::string& transactionId,
                                          bool writeCapable) {
      std::shared_ptr<coral::ISessionProxy> coralSession =
          createCoralSession(connectionString, transactionId, writeCapable);

      std::string principalName("");
      if (!m_authenticationService.empty()) {
        // need to hard-code somewhere the target name...
        if (m_authenticationService == "COND/Services/RelationalAuthenticationService") {
          coral::IHandle<coral::IAuthenticationService> authSvc =
              coral::Context::instance().query<coral::IAuthenticationService>(m_authenticationService);
          IDbAuthentication* dbAuth = dynamic_cast<IDbAuthentication*>(authSvc.get());
          principalName = dbAuth->principalName();
        }
      }

      return Session(std::make_shared<SessionImpl>(coralSession, connectionString, principalName));
    }

    Session ConnectionPool::createSession(const std::string& connectionString, bool writeCapable) {
      return createSession(connectionString, "", writeCapable);
    }

    Session ConnectionPool::createReadOnlySession(const std::string& connectionString,
                                                  const std::string& transactionId) {
      return createSession(connectionString, transactionId);
    }

    std::shared_ptr<coral::ISessionProxy> ConnectionPool::createCoralSession(const std::string& connectionString,
                                                                             bool writeCapable) {
      return createCoralSession(connectionString, "", writeCapable);
    }

    void ConnectionPool::setMessageVerbosity(coral::MsgLevel level) { m_messageLevel = level; }

    void ConnectionPool::setConnectionTimeout(int seconds) { m_connectionTimeout = seconds; }

    void ConnectionPool::setLogDestination(Logger& logger) { m_msgReporter->subscribe(logger); }

  }  // namespace persistency
}  // namespace cond
