#include "CondCore/CondDB/interface/FileUtils.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Auth.h"
#include "RelationalAccess/AuthenticationCredentials.h"
#include "CoralCommon/Cipher.h"
#include "RelationalAccess/AuthenticationServiceException.h"
#include "CoralKernel/IPropertyManager.h"
#include "CoralKernel/Property.h"
#include "CoralKernel/Context.h"
#include "CondCore/CondDB/interface/CoralServiceMacros.h"
#include "Utilities/Xerces/interface/Xerces.h"
#include "xercesc/parsers/XercesDOMParser.hpp"
#include "xercesc/framework/MemBufInputSource.hpp"
#include "xercesc/dom/DOM.hpp"
#include "xercesc/sax/HandlerBase.hpp"
#include "xercesc/util/XMLString.hpp"
#include "xercesc/util/PlatformUtils.hpp"
#include "XMLAuthenticationService.h"

#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sys/stat.h>

#include "CoralBase/MessageStream.h"

constexpr char XML_AUTHENTICATION_FILE[] = "authentication.xml";

cond::XMLAuthenticationService::DataSourceEntry::DataSourceEntry(const std::string& serviceName,
                                                                 const std::string& connectionName)
    : m_serviceName(serviceName),
      m_connectionName(connectionName),
      m_default(new coral::AuthenticationCredentials(serviceName)),
      m_data() {}

cond::XMLAuthenticationService::DataSourceEntry::~DataSourceEntry() {
  delete m_default;
  for (std::map<std::string, coral::AuthenticationCredentials*>::iterator iData = m_data.begin(); iData != m_data.end();
       ++iData)
    delete iData->second;
}

void cond::XMLAuthenticationService::DataSourceEntry::appendCredentialItem(const std::string& item,
                                                                           const std::string& value) {
  m_default->registerItem(item, value);
}

void cond::XMLAuthenticationService::DataSourceEntry::appendCredentialItemForRole(const std::string& item,
                                                                                  const std::string& value,
                                                                                  const std::string& role) {
  std::map<std::string, coral::AuthenticationCredentials*>::iterator iRole = m_data.find(role);
  if (iRole == m_data.end()) {
    iRole = m_data.insert(std::make_pair(role, new coral::AuthenticationCredentials(m_serviceName))).first;
  }
  iRole->second->registerItem(item, value);
}

const coral::IAuthenticationCredentials& cond::XMLAuthenticationService::DataSourceEntry::credentials() const {
  return *m_default;
}

const coral::IAuthenticationCredentials& cond::XMLAuthenticationService::DataSourceEntry::credentials(
    const std::string& role) const {
  /**
  std::map< std::string, coral::AuthenticationCredentials* >::const_iterator iRole = m_data.find( role );
  if ( iRole == m_data.end() )
    throw coral::UnknownRoleException( m_serviceName,
                                       m_connectionName,
                                       role );
  return *( iRole->second );
  **/
  return *m_default;
}

cond::XMLAuthenticationService::XMLAuthenticationService::XMLAuthenticationService(const std::string& key)
    : coral::Service(key), m_isInitialized(false), m_inputFileName(""), m_data(), m_mutexLock(), m_callbackID(0) {
  boost::function1<void, std::string> cb(std::bind(
      &cond::XMLAuthenticationService::XMLAuthenticationService::setAuthenticationPath, this, std::placeholders::_1));

  coral::Property* pm = dynamic_cast<coral::Property*>(
      coral::Context::instance().PropertyManager().property(auth::COND_AUTH_PATH_PROPERTY));
  if (pm) {
    setAuthenticationPath(pm->get());
    m_callbackID = pm->registerCallback(cb);
  }
}

cond::XMLAuthenticationService::XMLAuthenticationService::~XMLAuthenticationService() {
  for (std::map<std::string, cond::XMLAuthenticationService::DataSourceEntry*>::iterator iConnection = m_data.begin();
       iConnection != m_data.end();
       ++iConnection)
    delete iConnection->second;
}

void cond::XMLAuthenticationService::XMLAuthenticationService::setAuthenticationPath(const std::string& inputPath) {
  std::filesystem::path AuthPath(inputPath);
  if (std::filesystem::is_directory(AuthPath)) {
    AuthPath /= std::filesystem::path(XML_AUTHENTICATION_FILE);
  }

  m_inputFileName = AuthPath.string();
  reset();
}

bool cond::XMLAuthenticationService::XMLAuthenticationService::processFile(const std::string& inputFileName) {
  coral::MessageStream log("cond::XMLAuthenticationService::processFile");
  //std::cout<< "Processing file \""<< inputFileName<<"\"" <<std::endl;
  bool result = true;

  cond::FileReader inputFile;
  std::string cont("");
  try {
    inputFile.read(inputFileName);
    cont = inputFile.content();
  } catch (const cond::Exception& exc) {
    log << coral::Error << "File \"" << inputFileName << "\" not found." << std::string(exc.what())
        << coral::MessageStream::endmsg;
    return false;
  }

  std::filesystem::path filePath(inputFileName);
  std::string name = filePath.filename().string();

  /**
  if(name!=XML_AUTHENTICATION_FILE){
    cond::DecodingKey key;
    try{
      key.readUserKeyString(cont);
      log << coral::Debug << "Decoding content of file \""<< key.dataSource()<<"\""<<coral::MessageStream::endmsg;
      cond::FileReader dataFile;
      dataFile.read(key.dataSource());
      cont = dataFile.content();
      cont = coral::Cipher::decode(cont,key.key());      
    } catch (const cond::Exception& exc){
      log << coral::Error << std::string(exc.what())<<coral::MessageStream::endmsg;
      return false;
    }
    
  } else {
    //seal::MessageStream log( this, this->name(),seal::Msg::Verbose );
    log<<coral::Debug<< "Authentication file is expected standard XML."<<coral::MessageStream::endmsg;
  }
  **/

  xercesc::MemBufInputSource* memBufInputSource = nullptr;

  try {
    xercesc::XercesDOMParser parser;
    parser.setValidationScheme(xercesc::XercesDOMParser::Val_Always);
    parser.setDoNamespaces(true);

    xercesc::HandlerBase errorHandler;
    parser.setErrorHandler(&errorHandler);

    const char* bufferId = "InMemoryDocument";
    const char* buffer = cont.c_str();

    memBufInputSource = new xercesc::MemBufInputSource((const XMLByte*)buffer, strlen(buffer), bufferId, false);

    parser.parse(*memBufInputSource);

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

          // Locate the credential
          cond::XMLAuthenticationService::DataSourceEntry* credential = nullptr;
          std::map<std::string, cond::XMLAuthenticationService::DataSourceEntry*>::iterator iConnection =
              m_data.find(sConnectionName);
          if (iConnection != m_data.end()) {
            credential = iConnection->second;
            // Issue a warning here.
            //coral::MessageStream log( this, this->name(),seal::Msg::Verbose );
            log << coral::Debug << "Credential parameters for connection string \"" << sConnectionName
                << "\" have already been defined. Only new elements are appended, while existing will be ignored."
                << coral::MessageStream::endmsg;
          } else {
            credential = new cond::XMLAuthenticationService::DataSourceEntry(this->name(), sConnectionName);
            m_data.insert(std::make_pair(sConnectionName, credential));
          }

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

                  credential->appendCredentialItem(sParameterName, sParameterValue);
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

                          credential->appendCredentialItemForRole(sRoleParameterName, sRoleParameterValue, sRoleName);
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
    //coral::MessageStream log( this, this->name(),coral::Msg::Verbose );
    //log << coral::Msg::Error << message << coral::flush;
    log << coral::Error << std::string(message) << coral::MessageStream::endmsg;
    xercesc::XMLString::release(&message);
    result = false;
  } catch (const xercesc::DOMException& toCatch) {
    char* message = xercesc::XMLString::transcode(toCatch.msg);
    //seal::MessageStream log( this, this->name(),seal::Msg::Verbose );
    //log << seal::Msg::Error << message << seal::flush;
    log << coral::Error << std::string(message) << coral::MessageStream::endmsg;
    xercesc::XMLString::release(&message);
    result = false;
  } catch (const xercesc::SAXException& toCatch) {
    char* message = xercesc::XMLString::transcode(toCatch.getMessage());
    //seal::MessageStream log( this, this->name(),seal::Msg::Verbose );
    //log << seal::Msg::Error << message << seal::flush;
    log << coral::Error << std::string(message) << coral::MessageStream::endmsg;
    xercesc::XMLString::release(&message);
    result = false;
  } catch (...) {
    //seal::MessageStream log( this, this->name(),seal::Msg::Verbose );
    //log << seal::Msg::Error << "Unexpected Exception parsing file \"" << inputFileName << "\"" << seal::flush;
    log << coral::Error << "Unexpected Exception parsing file \"" << inputFileName << "\""
        << coral::MessageStream::endmsg;
    result = false;
  }
  if (memBufInputSource)
    delete memBufInputSource;
  return result;
}

bool cond::XMLAuthenticationService::XMLAuthenticationService::initialize() {
  coral::MessageStream log("cond::XMLAuthenticationService::initialize");
  std::set<std::string> inputFileNames = this->verifyFileName();
  if (inputFileNames.empty()) {
    //seal::MessageStream log( this, this->name(),seal::Msg::Verbose );
    //std::cout<< "Could not open \"" << m_inputFileName << "\" for reading" << std::endl;
    log << coral::Debug << "Could not open \"" << m_inputFileName << "\" for reading" << coral::MessageStream::endmsg;
    return false;
  }

  try {
    cms::concurrency::xercesInitialize();
  } catch (const xercesc::XMLException& toCatch) {
    char* message = xercesc::XMLString::transcode(toCatch.getMessage());
    //seal::MessageStream log( this, this->name(),seal::Msg::Verbose );
    //log << seal::Msg::Error << message << seal::flush;
    log << coral::Error << std::string(message) << coral::MessageStream::endmsg;
    xercesc::XMLString::release(&message);
    return false;
  }

  bool result = false;
  for (std::set<std::string>::const_reverse_iterator iFileName = inputFileNames.rbegin();
       iFileName != inputFileNames.rend();
       ++iFileName) {
    if (this->processFile(*iFileName)) {
      result = true;
    }
  }

  cms::concurrency::xercesTerminate();

  m_isInitialized = result;
  if (!m_isInitialized)
    reset();
  return result;
}

void cond::XMLAuthenticationService::XMLAuthenticationService::reset() {
  for (std::map<std::string, cond::XMLAuthenticationService::DataSourceEntry*>::iterator iConnection = m_data.begin();
       iConnection != m_data.end();
       ++iConnection)
    delete iConnection->second;
  m_data.clear();
  m_isInitialized = false;
}

const coral::IAuthenticationCredentials& cond::XMLAuthenticationService::XMLAuthenticationService::credentials(
    const std::string& connectionString) const {
  boost::mutex::scoped_lock lock(m_mutexLock);
  if (!m_isInitialized) {
    const_cast<cond::XMLAuthenticationService::XMLAuthenticationService*>(this)->initialize();
  }
  std::map<std::string, cond::XMLAuthenticationService::DataSourceEntry*>::const_iterator iConnection =
      m_data.find(connectionString);
  if (iConnection == m_data.end())
    throw coral::UnknownConnectionException(this->name(), connectionString);
  return iConnection->second->credentials();
}

const coral::IAuthenticationCredentials& cond::XMLAuthenticationService::XMLAuthenticationService::credentials(
    const std::string& connectionString, const std::string& role) const {
  boost::mutex::scoped_lock lock(m_mutexLock);
  if (!m_isInitialized) {
    const_cast<cond::XMLAuthenticationService::XMLAuthenticationService*>(this)->initialize();
  }
  std::map<std::string, cond::XMLAuthenticationService::DataSourceEntry*>::const_iterator iConnection =
      m_data.find(connectionString);
  if (iConnection == m_data.end())
    throw coral::UnknownConnectionException(this->name(), connectionString);
  return iConnection->second->credentials(role);
}

std::set<std::string> cond::XMLAuthenticationService::XMLAuthenticationService::verifyFileName() {
  coral::MessageStream log("cond::XMLAuthenticationService::verifyFileName");
  std::set<std::string> fileNames;

  // Try the file name as is...
  std::filesystem::path filePath(m_inputFileName);
  if (std::filesystem::exists(m_inputFileName)) {
    if (std::filesystem::is_directory(m_inputFileName)) {
      //seal::MessageStream log( this, this->name(),seal::Msg::Verbose );
      log << coral::Error << "Provided path \"" << m_inputFileName << "\" is a directory."
          << coral::MessageStream::endmsg;
      return fileNames;
    }
    std::filesystem::path fullPath = filePath.lexically_normal();
    fileNames.insert(fullPath.string());
    if (filePath.is_absolute())
      return fileNames;
  }

  // Try to find other files in the path variable
  const char* thePathVariable = std::getenv("CORAL_AUTH_PATH");
  if (!thePathVariable)
    return fileNames;
  log << coral::Debug << "File \"" << m_inputFileName
      << "\" not found in the current directory. Trying in the search path." << coral::MessageStream::endmsg;

  std::string searchPath(thePathVariable);
  //std::cout<<"searchPath "<<searchPath<<std::endl;
  if (std::filesystem::exists(searchPath)) {
    if (!std::filesystem::is_directory(searchPath)) {
      log << coral::Debug << "Search path \"" << searchPath << "\" is not a directory." << coral::MessageStream::endmsg;
      return fileNames;
    }
    std::filesystem::path fullPath(searchPath);
    fullPath /= filePath;
    fileNames.insert(fullPath.string());
  } else {
    log << coral::Debug << "Search path \"" << searchPath << "\" does not exist." << coral::MessageStream::endmsg;
    return fileNames;
  }

  return fileNames;
}

DEFINE_CORALSERVICE(cond::XMLAuthenticationService::XMLAuthenticationService, "COND/Services/XMLAuthenticationService");
