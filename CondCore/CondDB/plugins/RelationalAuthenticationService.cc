#include "CondCore/CondDB/interface/Auth.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "RelationalAccess/AuthenticationCredentials.h"
#include "RelationalAccess/AuthenticationServiceException.h"
#include "CondCore/CondDB/interface/CoralServiceMacros.h"
#include "RelationalAuthenticationService.h"
//
#include "RelationalAccess/AuthenticationServiceException.h"
#include "CoralKernel/IPropertyManager.h"
#include "CoralKernel/Property.h"
#include "CoralKernel/Context.h"
//
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <sys/stat.h>

#include "CoralBase/MessageStream.h"

cond::RelationalAuthenticationService::RelationalAuthenticationService::RelationalAuthenticationService(
    const std::string& key)
    : coral::Service(key), m_authenticationPath(""), m_db(), m_cache(), m_callbackID(0) {
  boost::function1<void, std::string> cb(
      std::bind(&cond::RelationalAuthenticationService::RelationalAuthenticationService::setAuthenticationPath,
                this,
                std::placeholders::_1));

  coral::Property* pm = dynamic_cast<coral::Property*>(
      coral::Context::instance().PropertyManager().property(auth::COND_AUTH_PATH_PROPERTY));
  if (pm) {
    setAuthenticationPath(pm->get());
    m_callbackID = pm->registerCallback(cb);
  }
}

cond::RelationalAuthenticationService::RelationalAuthenticationService::~RelationalAuthenticationService() {}

void cond::RelationalAuthenticationService::RelationalAuthenticationService::setAuthenticationPath(
    const std::string& inputPath) {
  m_authenticationPath = inputPath;
  m_cache.reset();
}

const coral::IAuthenticationCredentials&
cond::RelationalAuthenticationService::RelationalAuthenticationService::credentials(
    const std::string& connectionStr) const {
  std::string connectionString = to_lower(connectionStr);
  const coral::IAuthenticationCredentials* creds = m_cache.get(connectionString);
  if (!creds) {
    std::string credsStoreConn = m_db.setUpForConnectionString(connectionString, m_authenticationPath);
    coral::MessageStream log("cond::RelationalAuthenticationService::credentials");
    log << coral::Debug << "Connecting to the credential repository in \"" << credsStoreConn << "\" with principal \""
        << m_db.keyPrincipalName() << "\"." << coral::MessageStream::endmsg;
    m_db.selectForUser(m_cache);
  }
  creds = m_cache.get(connectionString);
  if (!creds) {
    std::string msg("Connection to \"");
    msg += connectionString + "\"";
    msg += " with role \"COND_DEFAULT_ROLE\" is not available for ";
    msg += m_db.keyPrincipalName();
    cond::throwException(msg, "cond::RelationalAuthenticationService::RelationalAuthenticationService::credentials");
  }
  return *creds;
}

const coral::IAuthenticationCredentials&
cond::RelationalAuthenticationService::RelationalAuthenticationService::credentials(const std::string& connectionStr,
                                                                                    const std::string& role) const {
  std::string connectionString = to_lower(connectionStr);
  const coral::IAuthenticationCredentials* creds = m_cache.get(connectionString, role);
  if (!creds) {
    std::string credsStoreConn = m_db.setUpForConnectionString(connectionString, m_authenticationPath);
    coral::MessageStream log("cond::RelationalAuthenticationService::credentials");
    log << coral::Debug << "Connecting to the credential repository in \"" << credsStoreConn << "\" with principal \""
        << m_db.keyPrincipalName() << "\"." << coral::MessageStream::endmsg;
    m_db.selectForUser(m_cache);
  }
  creds = m_cache.get(connectionString, role);
  if (!creds) {
    std::string msg("Connection to \"");
    msg += connectionString + "\"";
    msg += " with role \"" + role + "\" is not available for ";
    msg += m_db.keyPrincipalName();
    cond::throwException(msg, "cond::RelationalAuthenticationService::RelationalAuthenticationService::credentials");
  }
  return *creds;
}

std::string cond::RelationalAuthenticationService::RelationalAuthenticationService::principalName() {
  return m_db.keyPrincipalName();
}

DEFINE_CORALSERVICE(cond::RelationalAuthenticationService::RelationalAuthenticationService,
                    "COND/Services/RelationalAuthenticationService");
