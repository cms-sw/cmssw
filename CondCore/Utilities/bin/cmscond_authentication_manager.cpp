#include "CondCore/CondDB/interface/Auth.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/CredentialStore.h"
#include "CondCore/Utilities/interface/Utilities.h"
//
#include "RelationalAccess/AuthenticationCredentials.h"
#include "Utilities/Xerces/interface/Xerces.h"
//
#include <iostream>
#include <fstream>
#include <iomanip>

namespace coral_bridge {
  bool parseXMLAuthenticationFile(const std::string& inputFileName, coral_bridge::AuthenticationCredentialSet& data);

  // class for creating the XML Authentication file
  class XMLAuthenticationFileContent {
  public:
    explicit XMLAuthenticationFileContent(std::ostream& out);

    bool openConnectionEntry(const std::string& pcs);

    bool closeConnectionEntry();

    bool openRoleEntry(const std::string& roleName);

    bool closeRoleEntry();

    bool addCredentialEntry(const std::string& userName, const std::string& password);

    void close();

  private:
    std::ostream& m_out;

    bool m_connectionListOpen;

    bool m_connectionEntryOpen;

    bool m_roleEntryOpen;

    unsigned int m_ind;
  };

}  // namespace coral_bridge

namespace cond {
  class AuthenticationManager : public Utilities {
  public:
    AuthenticationManager();
    ~AuthenticationManager() override;
    int execute() override;
  };
}  // namespace cond

cond::AuthenticationManager::AuthenticationManager() : Utilities("cmscond_authentication_manager") {
  addOption<std::string>("authPath", "P", "authentication path");
  addOption<bool>("create", "", "[-s a0 -u a1] create creds db");
  addOption<bool>("reset_admin", "", "[-s a0 -u a1] reset write credentials for admin");
  addOption<bool>("drop", "", "[-c a0 -u a1] drop creds db");
  addOption<bool>("update_princ", "", "[-s a0 -k a1 (-a)] add/update principal");
  addOption<bool>("update_conn", "", "[-s a0 -u a1 (-l a3)] add/update connection");
  addOption<bool>("set_perm", "", "[-s a0 -n a1 -r a2 -c a3 -l a4] set permission");
  addOption<bool>("unset_perm", "", "[-s a0 -n a1 -r a2 -c a3] unset permission");
  addOption<bool>("list_conn", "", "[-s arg ] list connections");
  addOption<bool>("list_perm", "", "[-s a0 (-n a1)(-r a2)(-c a3)] list permissions");
  addOption<bool>("list_princ", "", "[-s arg] list principals");
  addOption<bool>("remove_princ", "", "[-s a0 -n a1] remove principal");
  addOption<bool>("remove_conn", "", "[-s a0 -l a1] remove connection");
  addOption<bool>("admin", "a", "add admin privileges");
  addOption<bool>("force_update", "f", "force update");
  addOption<std::string>("import", "", "[a0 -s a1 -n a2 (-f)] import from the xml file");
  addOption<std::string>("export", "", "[a0 -s a1] export to the xml file");
  addOption<std::string>("service", "s", "service name");
  addOption<std::string>("princ_name", "n", "the principal");
  addOption<std::string>("key", "k", "the key filename");
  addOption<std::string>("role", "r", "the role");
  addOption<std::string>("connectionString", "c", "the connection string");
  addOption<std::string>("connectionLabel", "l", "the connection label");
  addOption<std::string>("userName", "u", "the user name");
}

cond::AuthenticationManager::~AuthenticationManager() {}

int cond::AuthenticationManager::execute() {
  if (hasDebug())
    coral::MessageStream::setMsgVerbosity(coral::Debug);
  std::string authPath("");
  if (hasOptionValue("authPath"))
    authPath = getOptionValue<std::string>("authPath");
  if (authPath.empty()) {
    const char* authEnv = ::getenv(auth::COND_AUTH_PATH);
    if (authEnv) {
      authPath += authEnv;
    } else {
      authEnv = ::getenv("HOME");
      if (authEnv) {
        authPath += authEnv;
      }
    }
  }

  bool drop = hasOptionValue("drop");
  bool create = hasOptionValue("create");
  bool reset_admin = hasOptionValue("reset_admin");
  bool update_princ = hasOptionValue("update_princ");
  bool update_conn = hasOptionValue("update_conn");
  bool list_conn = hasOptionValue("list_conn");
  bool list_perm = hasOptionValue("list_perm");
  bool list_princ = hasOptionValue("list_princ");
  bool remove_princ = hasOptionValue("remove_princ");
  bool remove_conn = hasOptionValue("remove_conn");
  bool set_perm = hasOptionValue("set_perm");
  bool unset_perm = hasOptionValue("unset_perm");
  bool import = hasOptionValue("import");
  bool exp = hasOptionValue("export");

  bool not_exec = true;
  CredentialStore credDb;

  if (drop) {
    std::string connectionString = getOptionValue<std::string>("connectionString");
    std::string userName = getOptionValue<std::string>("userName");
    std::string password = cond::getpassForUser(userName);
    credDb.drop(connectionString, userName, password);
    std::cout << "Credential Repository in " << connectionString << " has been dropped." << std::endl;
    not_exec = false;
  }

  std::string service("");
  if (create && not_exec) {
    service = getOptionValue<std::string>("service");
    std::string userName = getOptionValue<std::string>("userName");
    std::string password = cond::getpassForUser(userName);
    std::string credsStore = credDb.setUpForService(service, authPath);
    credDb.createSchema(credsStore, userName, password);
    std::cout << "Credential Repository for service " << service << " has been created." << std::endl;
    not_exec = false;
  }

  if (reset_admin || update_princ || update_conn || remove_princ || remove_conn || set_perm || unset_perm || import ||
      list_conn || list_princ || list_perm || exp) {
    service = getOptionValue<std::string>("service");
    std::string credsStore = credDb.setUpForService(service, authPath);
    std::cout << "Authenticated principal " << credDb.keyPrincipalName() << std::endl;
    std::cout << "Connecting with credential repository in \"" << credsStore << "\"" << std::endl;
  }

  if (reset_admin && not_exec) {
    std::string userName = getOptionValue<std::string>("userName");
    std::string password = cond::getpassForUser(userName);
    credDb.resetAdmin(userName, password);
    std::cout << "Admin access to service " << credDb.serviceName() << " has been reset." << std::endl;
    not_exec = false;
  }

  if (update_princ && not_exec) {
    bool adminOpt = hasOptionValue("admin");
    std::string key = getOptionValue<std::string>("key");
    auth::DecodingKey pk;
    pk.init(key, auth::COND_KEY);
    credDb.updatePrincipal(pk.principalName(), pk.principalKey(), adminOpt);
    std::cout << "Principal " << pk.principalName() << " has been updated." << std::endl;
    not_exec = false;
  }

  if (update_conn && not_exec) {
    std::string userName = getOptionValue<std::string>("userName");
    std::string password = cond::getpassForUser(userName);
    std::string connectionLabel = schemaLabel(service, userName);
    if (hasOptionValue("connectionLabel")) {
      connectionLabel = getOptionValue<std::string>("connectionLabel");
    }
    credDb.updateConnection(connectionLabel, userName, password);
    std::cout << "Credentials for connection " << connectionLabel << " have been updated." << std::endl;
    not_exec = false;
  }

  if (remove_princ && not_exec) {
    std::string principal = getOptionValue<std::string>("princ_name");
    credDb.removePrincipal(principal);
    std::cout << "Principal " << principal << " has been removed." << std::endl;
    not_exec = false;
  }

  if (remove_conn && not_exec) {
    std::string connectionLabel = getOptionValue<std::string>("connectionLabel");
    credDb.removeConnection(connectionLabel);
    std::cout << "Connection " << connectionLabel << " has been removed." << std::endl;
    not_exec = false;
  }

  if (set_perm && not_exec) {
    std::string principal = getOptionValue<std::string>("princ_name");
    std::string role = getOptionValue<std::string>("role");
    std::string connectionString = getOptionValue<std::string>("connectionString");
    std::string connectionLabel = getOptionValue<std::string>("connectionLabel");
    credDb.setPermission(principal, role, connectionString, connectionLabel);
    std::cout << "Permission for principal " << principal << " to access resource " << connectionString << " with role "
              << role << " has been set." << std::endl;
    not_exec = false;
  }

  if (unset_perm && not_exec) {
    std::string connectionString = getOptionValue<std::string>("connectionString");
    std::string principal("");
    if (hasOptionValue("princ_name"))
      principal = getOptionValue<std::string>("princ_name");
    std::string role("");
    if (hasOptionValue("role"))
      role = getOptionValue<std::string>("role");
    size_t n = credDb.unsetPermission(principal, role, connectionString);
    if (n) {
      std::cout << n << " permissions to access resource " << connectionString;
      if (!role.empty())
        std::cout << " with role " << role;
      std::cout << " have been unset";
      if (!principal.empty())
        std::cout << " for principal " << principal << " ";
    } else {
      std::cout << "No Permission found to access resource " << connectionString;
      if (!role.empty())
        std::cout << " with role " << role;
      if (!principal.empty())
        std::cout << " for principal " << principal << " ";
    }
    std::cout << "." << std::endl;
    not_exec = false;
  }

  if (import && not_exec) {
    bool forceUp = hasOptionValue("force_update");
    std::string fileName = getOptionValue<std::string>("import");
    std::string principal = getOptionValue<std::string>("princ_name");
    coral_bridge::AuthenticationCredentialSet source;
    if (!coral_bridge::parseXMLAuthenticationFile(fileName, source)) {
      std::cout << "Error: XML parsing failed." << std::endl;
      return 1;
    }
    std::cout << "Importing " << source.data().size() << " connection items." << std::endl;
    credDb.importForPrincipal(principal, source, forceUp);
    std::cout << "Credential set for principal " << principal << " has been imported from file " << fileName
              << std::endl;
    not_exec = false;
  }

  if (list_conn && not_exec) {
    std::map<std::string, std::pair<std::string, std::string> > data;
    credDb.listConnections(data);
    std::cout << "Found " << data.size() << " connection(s)." << std::endl;
    std::cout << std::endl;
    static std::string connectionLabelH("connection label");
    static std::string userNameH("username");
    static std::string passwordH("password");
    size_t connectionLabelW = connectionLabelH.size();
    size_t userNameW = 0;
    size_t passwordW = 0;
    for (std::map<std::string, std::pair<std::string, std::string> >::const_iterator iC = data.begin();
         iC != data.end();
         ++iC) {
      const std::string& userName = iC->second.first;
      const std::string& password = iC->second.second;
      const std::string& connectionLabel = iC->first;
      if (connectionLabelW < connectionLabel.size())
        connectionLabelW = connectionLabel.size();
      if (userNameW < userName.size())
        userNameW = userName.size();
      if (passwordW < password.size())
        passwordW = password.size();
    }
    if (userNameW > 0 && userNameW < userNameH.size())
      userNameW = userNameH.size();
    if (passwordW > 0 && passwordW < passwordH.size())
      passwordW = passwordH.size();
    std::cout << std::setiosflags(std::ios_base::left);
    std::cout << std::setw(connectionLabelW) << connectionLabelH;
    if (userNameW) {
      std::cout << "  " << std::setw(userNameW) << userNameH;
    }
    if (passwordW) {
      std::cout << "  " << std::setw(passwordW) << passwordH;
    }
    std::cout << std::endl;
    std::cout << std::setfill('-');
    std::cout << std::setw(connectionLabelW) << "";
    if (userNameW) {
      std::cout << "  " << std::setw(userNameW) << "";
    }
    if (passwordW) {
      std::cout << "  " << std::setw(passwordW) << "";
    }
    std::cout << std::endl;
    std::cout << std::setfill(' ');
    for (std::map<std::string, std::pair<std::string, std::string> >::const_iterator iC = data.begin();
         iC != data.end();
         ++iC) {
      const std::string& connectionLabel = iC->first;
      const std::string& userName = iC->second.first;
      const std::string& password = iC->second.second;
      std::cout << std::setw(connectionLabelW) << connectionLabel << "  " << std::setw(userNameW) << userName << "  "
                << std::setw(passwordW) << password << std::endl;
    }
    not_exec = false;
  }

  if (list_princ && not_exec) {
    std::vector<std::string> data;
    credDb.listPrincipals(data);
    std::cout << "Found " << data.size() << " principal(s)." << std::endl;
    std::cout << std::endl;
    static std::string principalH("principal name");
    size_t principalW = principalH.size();
    for (std::vector<std::string>::const_iterator iP = data.begin(); iP != data.end(); ++iP) {
      const std::string& principal = *iP;
      if (principalW < principal.size())
        principalW = principal.size();
    }
    std::cout << std::setiosflags(std::ios_base::left);
    std::cout << std::setw(principalW) << principalH << std::endl;
    std::cout << std::setfill('-');
    std::cout << std::setw(principalW) << "" << std::endl;
    std::cout << std::setfill(' ');
    for (std::vector<std::string>::const_iterator iP = data.begin(); iP != data.end(); ++iP) {
      std::cout << std::setw(principalW) << *iP << std::endl;
    }
    not_exec = false;
  }

  if (list_perm && not_exec) {
    std::vector<CredentialStore::Permission> data;
    std::string pName("");
    std::string role("");
    std::string conn("");
    if (hasOptionValue("connectionString"))
      conn = getOptionValue<std::string>("connectionString");
    if (hasOptionValue("princ_name"))
      pName = getOptionValue<std::string>("princ_name");
    if (hasOptionValue("role"))
      role = getOptionValue<std::string>("role");
    credDb.selectPermissions(pName, role, conn, data);
    std::cout << "Found " << data.size() << " permission(s)." << std::endl;
    std::cout << std::endl;
    static std::string connectionStringH("connection string");
    static std::string principalH("principal name");
    static std::string roleH("role");
    static std::string connectionLabelH("connection label");
    size_t connectionStringW = connectionStringH.size();
    size_t principalW = principalH.size();
    size_t roleW = roleH.size();
    size_t connectionLabelW = connectionLabelH.size();
    for (std::vector<CredentialStore::Permission>::const_iterator iP = data.begin(); iP != data.end(); ++iP) {
      const std::string& connectionString = iP->connectionString;
      if (connectionStringW < connectionString.size())
        connectionStringW = connectionString.size();
      const std::string& principal = iP->principalName;
      if (principalW < principal.size())
        principalW = principal.size();
      const std::string& role = iP->role;
      if (roleW < role.size())
        roleW = role.size();
      const std::string& connectionLabel = iP->connectionLabel;
      if (connectionLabelW < connectionLabel.size())
        connectionLabelW = connectionLabel.size();
    }
    std::cout << std::setiosflags(std::ios_base::left);
    std::cout << std::setw(connectionStringW) << connectionStringH << "  " << std::setw(principalW) << principalH
              << "  ";
    std::cout << std::setw(roleW) << roleH << "  " << std::setw(connectionLabelW) << connectionLabelH << std::endl;
    std::cout << std::setfill('-');
    std::cout << std::setw(connectionStringW) << ""
              << "  " << std::setw(principalW) << ""
              << "  " << std::setw(roleW) << ""
              << "  " << std::setw(connectionLabelW) << "" << std::endl;
    std::cout << std::setfill(' ');
    for (std::vector<CredentialStore::Permission>::const_iterator iP = data.begin(); iP != data.end(); ++iP) {
      std::cout << std::setw(connectionStringW) << iP->connectionString << "  " << std::setw(principalW)
                << iP->principalName << "  ";
      ;
      std::cout << std::setw(roleW) << iP->role << "  " << std::setw(connectionLabelW) << iP->connectionLabel
                << std::endl;
    }
    not_exec = false;
  }

  if (exp && not_exec) {
    std::string fileName = getOptionValue<std::string>("export");
    coral_bridge::AuthenticationCredentialSet data;
    credDb.exportAll(data);
    std::ofstream outFile(fileName);
    coral_bridge::XMLAuthenticationFileContent xmlFile(outFile);
    const std::map<std::pair<std::string, std::string>, coral::AuthenticationCredentials*>& creds = data.data();
    std::set<std::string> connections;
    bool started = false;
    for (std::map<std::pair<std::string, std::string>, coral::AuthenticationCredentials*>::const_iterator iEntry =
             creds.begin();
         iEntry != creds.end();
         iEntry++) {
      const std::string& connectStr = iEntry->first.first;
      std::set<std::string>::iterator iConn = connections.find(connectStr);
      if (iConn == connections.end()) {
        if (started)
          xmlFile.closeConnectionEntry();
        xmlFile.openConnectionEntry(connectStr);
        started = true;
        connections.insert(connectStr);
        std::pair<std::string, std::string> defRoleKey(connectStr, auth::COND_DEFAULT_ROLE);
        std::map<std::pair<std::string, std::string>, coral::AuthenticationCredentials*>::const_iterator iDef =
            creds.find(defRoleKey);
        if (iDef != creds.end()) {
          xmlFile.addCredentialEntry(iDef->second->valueForItem(coral::IAuthenticationCredentials::userItem()),
                                     iDef->second->valueForItem(coral::IAuthenticationCredentials::passwordItem()));
        }
      }
      const std::string& role = iEntry->first.second;
      if (role != auth::COND_DEFAULT_ROLE) {
        xmlFile.openRoleEntry(role);
        xmlFile.addCredentialEntry(iEntry->second->valueForItem(coral::IAuthenticationCredentials::userItem()),
                                   iEntry->second->valueForItem(coral::IAuthenticationCredentials::passwordItem()));
        xmlFile.closeRoleEntry();
      }
    }
    xmlFile.close();
    not_exec = false;
  }

  if (not_exec) {
    std::cout << "ERROR: no command specified." << std::endl;
    return 1;
  }
  if (hasDebug()) {
    std::cout << "LOGS from the Credential Store:" << std::endl;
    std::cout << credDb.log() << std::endl;
  }
  return 0;
}

int main(int argc, char** argv) {
  cond::AuthenticationManager mgr;
  return mgr.run(argc, argv);
}

#include "xercesc/parsers/XercesDOMParser.hpp"
#include "xercesc/dom/DOM.hpp"
#include "xercesc/sax/HandlerBase.hpp"
#include "xercesc/util/XMLString.hpp"
#include "xercesc/util/PlatformUtils.hpp"

#if defined __SUNPRO_CC
#pragma enable_warn
#elif defined _MSC_VER
#pragma warning(pop)
#endif

bool coral_bridge::parseXMLAuthenticationFile(const std::string& inputFileName, AuthenticationCredentialSet& data) {
  try {
    cms::concurrency::xercesInitialize();
  } catch (const xercesc::XMLException& toCatch) {
    char* message = xercesc::XMLString::transcode(toCatch.getMessage());
    //coral::MessageStream log( serviceName );
    //log << coral::Error << message << coral::MessageStream::endmsg;
    xercesc::XMLString::release(&message);
    return false;
  }

  bool result = true;
  try {
    xercesc::XercesDOMParser parser;
    parser.setValidationScheme(xercesc::XercesDOMParser::Val_Always);
    parser.setDoNamespaces(true);

    xercesc::HandlerBase errorHandler;
    parser.setErrorHandler(&errorHandler);

    parser.parse(inputFileName.c_str());

    xercesc::DOMDocument* document = parser.getDocument();

    XMLCh tempStr[20];
    xercesc::XMLString::transcode("connection", tempStr, 19);

    xercesc::DOMNodeList* connectionList = document->getElementsByTagName(tempStr);

    if (connectionList) {
      XMLSize_t numberOfConnections = connectionList->getLength();

      for (XMLSize_t iConnection = 0; iConnection < numberOfConnections; ++iConnection) {
        xercesc::DOMNode* connectionNode = connectionList->item(iConnection);

        if (connectionNode) {
          char* connectionName =
              xercesc::XMLString::transcode(connectionNode->getAttributes()->item(0)->getNodeValue());
          std::string sConnectionName = connectionName;
          xercesc::XMLString::release(&connectionName);

          xercesc::DOMNodeList* parameterList = connectionNode->getChildNodes();

          if (parameterList) {
            XMLSize_t numberOfParameters = parameterList->getLength();

            for (XMLSize_t iParameter = 0; iParameter < numberOfParameters; ++iParameter) {
              xercesc::DOMNode* parameterNode = parameterList->item(iParameter);

              if (parameterNode && parameterNode->getNodeType() == xercesc::DOMNode::ELEMENT_NODE) {
                char* nodeName = xercesc::XMLString::transcode(parameterNode->getNodeName());
                std::string sNodeName = nodeName;
                xercesc::XMLString::release(&nodeName);

                if (sNodeName == "parameter") {  // The default parameters
                  char* parameterName =
                      xercesc::XMLString::transcode(parameterNode->getAttributes()->item(0)->getNodeValue());
                  std::string sParameterName = parameterName;
                  xercesc::XMLString::release(&parameterName);
                  char* parameterValue =
                      xercesc::XMLString::transcode(parameterNode->getAttributes()->item(1)->getNodeValue());
                  std::string sParameterValue = parameterValue;
                  xercesc::XMLString::release(&parameterValue);

                  data.registerItem(sConnectionName, sParameterName, sParameterValue);
                } else if (sNodeName == "role") {  // A role
                  char* roleName =
                      xercesc::XMLString::transcode(parameterNode->getAttributes()->item(0)->getNodeValue());
                  std::string sRoleName = roleName;
                  xercesc::XMLString::release(&roleName);

                  // Retrieve the parameters for the role
                  xercesc::DOMNodeList* roleParameterList = parameterNode->getChildNodes();

                  if (roleParameterList) {
                    XMLSize_t numberOfRoleParameters = roleParameterList->getLength();

                    for (XMLSize_t iRoleParameter = 0; iRoleParameter < numberOfRoleParameters; ++iRoleParameter) {
                      xercesc::DOMNode* roleParameterNode = roleParameterList->item(iRoleParameter);
                      if (roleParameterNode && roleParameterNode->getNodeType() == xercesc::DOMNode::ELEMENT_NODE) {
                        char* roleNodeName = xercesc::XMLString::transcode(roleParameterNode->getNodeName());
                        std::string sRoleNodeName = roleNodeName;
                        xercesc::XMLString::release(&roleNodeName);

                        if (sRoleNodeName == "parameter") {
                          char* roleParameterName = xercesc::XMLString::transcode(
                              roleParameterNode->getAttributes()->item(0)->getNodeValue());
                          std::string sRoleParameterName = roleParameterName;
                          xercesc::XMLString::release(&roleParameterName);
                          char* roleParameterValue = xercesc::XMLString::transcode(
                              roleParameterNode->getAttributes()->item(1)->getNodeValue());
                          std::string sRoleParameterValue = roleParameterValue;
                          xercesc::XMLString::release(&roleParameterValue);

                          data.registerItem(sConnectionName, sRoleName, sRoleParameterName, sRoleParameterValue);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    parser.reset();
  } catch (const xercesc::XMLException& toCatch) {
    char* message = xercesc::XMLString::transcode(toCatch.getMessage());
    //coral::MessageStream log( serviceName );
    //log << coral::Error << message << coral::MessageStream::endmsg;
    xercesc::XMLString::release(&message);
    result = false;
  } catch (const xercesc::DOMException& toCatch) {
    char* message = xercesc::XMLString::transcode(toCatch.msg);
    //coral::MessageStream log( serviceName );
    //log << coral::Error << message << coral::MessageStream::endmsg;
    xercesc::XMLString::release(&message);
    result = false;
  } catch (...) {
    //coral::MessageStream log( serviceName );
    //log << coral::Error << "Unexpected Exception parsing file \"" << inputFileName << "\"" << coral::MessageStream::endmsg;
    result = false;
  }

  cms::concurrency::xercesTerminate();

  return result;
}

#include "RelationalAccess/IAuthenticationCredentials.h"
#include <iomanip>

coral_bridge::XMLAuthenticationFileContent::XMLAuthenticationFileContent(std::ostream& out)
    : m_out(out), m_connectionListOpen(false), m_connectionEntryOpen(false), m_roleEntryOpen(false), m_ind(0) {
  m_out << "<?xml version=\"1.0\" ?>" << std::endl;
  m_out << "<connectionlist>" << std::endl;
  m_connectionListOpen = true;
}

bool coral_bridge::XMLAuthenticationFileContent::openConnectionEntry(const std::string& pcs) {
  bool ret = false;
  if (m_connectionListOpen && !m_connectionEntryOpen) {
    m_out << std::endl;
    m_ind += 2;
    m_out << std::setw(m_ind) << "";
    m_out << "<connection name=\"" << pcs << "\" >" << std::endl;
    m_connectionEntryOpen = true;
    ret = true;
  }
  return ret;
}

bool coral_bridge::XMLAuthenticationFileContent::closeConnectionEntry() {
  bool ret = false;
  if (m_connectionEntryOpen) {
    m_out << std::setw(m_ind) << "";
    m_out << "</connection>" << std::endl;
    m_ind -= 2;
    ret = true;
    m_connectionEntryOpen = false;
  }
  return ret;
}

bool coral_bridge::XMLAuthenticationFileContent::openRoleEntry(const std::string& roleName) {
  bool ret = false;
  if (m_connectionEntryOpen && !m_roleEntryOpen) {
    m_ind += 2;
    m_out << std::setw(m_ind) << "";
    m_out << "<role name=\"" << roleName << "\" >" << std::endl;
    m_roleEntryOpen = true;
    ret = true;
  }
  return ret;
}

bool coral_bridge::XMLAuthenticationFileContent::closeRoleEntry() {
  bool ret = false;
  if (m_roleEntryOpen) {
    m_out << std::setw(m_ind) << "";
    m_out << "</role>" << std::endl;
    m_ind -= 2;
    ret = true;
    m_roleEntryOpen = false;
  }
  return ret;
}

bool coral_bridge::XMLAuthenticationFileContent::addCredentialEntry(const std::string& userName,
                                                                    const std::string& password) {
  bool ret = false;
  if (m_connectionEntryOpen) {
    m_out << std::setw(m_ind + 2) << "";
    m_out << "<parameter name=\"" << coral::IAuthenticationCredentials::userItem() << "\" value=\"" << userName
          << "\" />" << std::endl;
    m_out << std::setw(m_ind + 2) << "";
    m_out << "<parameter name=\"" << coral::IAuthenticationCredentials::passwordItem() << "\" value=\"" << password
          << "\" />" << std::endl;
    ret = true;
  }
  return ret;
}

void coral_bridge::XMLAuthenticationFileContent::close() {
  if (m_connectionListOpen) {
    if (m_connectionEntryOpen) {
      if (m_roleEntryOpen) {
        closeRoleEntry();
      }
      closeConnectionEntry();
    }
    m_out << std::endl;
    m_out << "</connectionlist>" << std::endl;
  }
  m_connectionListOpen = false;
}
