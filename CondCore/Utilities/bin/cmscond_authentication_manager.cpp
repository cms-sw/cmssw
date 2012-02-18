#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/CredentialStore.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/Utilities/interface/Utilities.h"
//
#include "RelationalAccess/AuthenticationCredentials.h"
//
#include <iostream>
#include <fstream>
#include <map>

namespace coral_bridge {
  bool parseXMLAuthenticationFile( const std::string& inputFileName, 
				   coral_bridge::AuthenticationCredentialSet& data);

  // class for creating the XML Authentication file
  class XMLAuthenticationFileContent {

  public:

    explicit XMLAuthenticationFileContent(std::ostream& out);

    bool openConnectionEntry(const std::string& pcs);

    bool closeConnectionEntry();

    bool openRoleEntry(const std::string& roleName);

    bool closeRoleEntry();

    bool addCredentialEntry(const std::string& userName, const std::string& password );

    void close();

  private:

    std::ostream& m_out;

    bool m_connectionListOpen;

    bool m_connectionEntryOpen;

    bool m_roleEntryOpen;

    unsigned int m_ind;
  };

}

namespace cond {
  class AuthenticationManager : public Utilities {
    public:
      AuthenticationManager();
      ~AuthenticationManager();
      int execute();
  };
}

cond::AuthenticationManager::AuthenticationManager():Utilities("cmscond_schema_manager"){
  addOption<bool>("create","c","create credential database");
  addOption<bool>("drop","d", "drop credential database");
  addOption<bool>("update","u", "insert or update a credential entry");
  addOption<bool>("list","l","list available credentials");
  addOption<bool>("remove","e","remove a credential entry");
  addOption<std::string>("service","n","service name");
  addOption<std::string>("principal","p","the principal");
  addOption<std::string>("role","r","the role");
  addOption<std::string>("connectionString","s","the connection string");
  addOption<std::string>("userName","a","the user name");
  addOption<std::string>("password","b","the password");
  addOption<std::string>("import","i","import from the specified xml file");
  addOption<std::string>("export","x","export to the specified xml file");
}

cond::AuthenticationManager::~AuthenticationManager(){
}

int cond::AuthenticationManager::execute(){
  bool drop= hasOptionValue("drop");
  bool create= hasOptionValue("create");
  bool update= hasOptionValue("update");
  //bool list = hasOptionValue("list");
  bool remove = hasOptionValue("remove");
  std::string service("");
  if( hasOptionValue("service") ) service = getOptionValue<std::string>("service");

  CredentialStore credDb;

  if( drop ){
    credDb.setUpForService( service );
    credDb.drop();
    return 0;
  }

  if( create ){
    credDb.setUpForService( service );
    credDb.createSchema();
    return 0;
  }

  if( update ){
    credDb.setUpForService( service );
    std::string principal = getOptionValue<std::string>("principal");
    std::string role = getOptionValue<std::string>("role");
    std::string connectionString = getOptionValue<std::string>("connectionString");
    std::string userName = getOptionValue<std::string>("userName");
    std::string password = getOptionValue<std::string>("password");
    credDb.update( principal, role, connectionString, userName, password );
    return 0;
  }

  if( remove ){
    credDb.setUpForService( service );
    std::string principal = getOptionValue<std::string>("principal");
    std::string role = getOptionValue<std::string>("role");
    std::string connectionString = getOptionValue<std::string>("connectionString");
    credDb.remove( principal, role, connectionString );
    return 0;
  }

  if( hasOptionValue("export") ){
    std::string principal("");
    if( hasOptionValue("principal")) principal = getOptionValue<std::string>("principal");
    std::string fileName = getOptionValue<std::string>("export");
    credDb.setUpForService( service );
    coral_bridge::AuthenticationCredentialSet data;
    if( principal.empty() ){
      credDb.exportAll( data );
    } else {
      credDb.exportForPrincipal( principal, data );
    }
    std::ofstream outFile( fileName );
    coral_bridge::XMLAuthenticationFileContent xmlFile( outFile );
    const std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >& creds = data.data();
    std::set<std::string> connections;
    bool started = false;
    for( std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >::const_iterator iEntry = creds.begin();
	 iEntry != creds.end(); iEntry++ ){
      const std::string& connectStr = iEntry->first.first;
      std::set<std::string>::iterator iConn = connections.find( connectStr );
      if( iConn == connections.end() ){
	if( started ) xmlFile.closeConnectionEntry();
	xmlFile.openConnectionEntry( connectStr );
	started = true;
	connections.insert( connectStr );
	std::pair<std::string,std::string> defRoleKey(connectStr,coral_bridge::AuthenticationCredentialSet::DEFAULT_ROLE);
	std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >::const_iterator iDef = creds.find( defRoleKey );
	if( iDef != creds.end() ){
	  xmlFile.addCredentialEntry( iDef->second->valueForItem( coral::IAuthenticationCredentials::userItem() ), 
				      iDef->second->valueForItem( coral::IAuthenticationCredentials::passwordItem() ) );
	}
      }
      const std::string& role = iEntry->first.second;
      if( role != coral_bridge::AuthenticationCredentialSet::DEFAULT_ROLE ){
	xmlFile.openRoleEntry( role );
	xmlFile.addCredentialEntry( iEntry->second->valueForItem( coral::IAuthenticationCredentials::userItem() ), 
				    iEntry->second->valueForItem( coral::IAuthenticationCredentials::passwordItem() ) );
	xmlFile.closeRoleEntry();
      }
    }
    xmlFile.close();
    return 0;
  }


  if( hasOptionValue("import") ){
    std::string principal("");
    if( hasOptionValue("principal")) principal = getOptionValue<std::string>("principal");
    std::string fileName = getOptionValue<std::string>("import");
    coral_bridge::AuthenticationCredentialSet source;
    if( !coral_bridge::parseXMLAuthenticationFile( fileName, source ) ){
      return 1;
    }
    credDb.setUpForService( service );
    credDb.importForPrincipal( principal, source );
    return 0;
  }

  /**
  if( list ){
    credDb.setUpForService( service );
    coral_bridge::AuthenticationCredentialSet data;
    if( principal.empty() ){
      credDb.exportAll( data );
    } else {
      credDb.exportForPrincipal( principal, data );
    }
    
  }
  **/

  return 0;
}

int main( int argc, char** argv ){

  cond::AuthenticationManager mgr;
  return mgr.run(argc,argv);
}

#include "xercesc/parsers/XercesDOMParser.hpp"
#include "xercesc/dom/DOM.hpp"
#include "xercesc/sax/HandlerBase.hpp"
#include "xercesc/util/XMLString.hpp"
#include "xercesc/util/PlatformUtils.hpp"

#if defined __SUNPRO_CC
#  pragma enable_warn
#elif defined _MSC_VER
#  pragma warning(pop)
#endif

bool coral_bridge::parseXMLAuthenticationFile( const std::string& inputFileName, 
					       AuthenticationCredentialSet& data){
  try
  {
    xercesc::XMLPlatformUtils::Initialize();
  }
  catch ( const xercesc::XMLException& toCatch )
  {
    char* message = xercesc::XMLString::transcode( toCatch.getMessage() );
    //coral::MessageStream log( serviceName );
    //log << coral::Error << message << coral::MessageStream::endmsg;
    xercesc::XMLString::release( &message );
    return false;
  }

  bool result = true;
  try
  {
    xercesc::XercesDOMParser parser;
    parser.setValidationScheme( xercesc::XercesDOMParser::Val_Always );
    parser.setDoNamespaces( true );

    xercesc::HandlerBase errorHandler;
    parser.setErrorHandler( &errorHandler );

    parser.parse( inputFileName.c_str() );

    xercesc::DOMDocument* document = parser.getDocument();

    XMLCh tempStr[20];
    xercesc::XMLString::transcode( "connection", tempStr, 19);

    xercesc::DOMNodeList* connectionList = document->getElementsByTagName( tempStr );

    if ( connectionList )
    {
      XMLSize_t numberOfConnections = connectionList->getLength();

      for ( XMLSize_t iConnection = 0; iConnection < numberOfConnections; ++iConnection )
      {
        xercesc::DOMNode* connectionNode = connectionList->item( iConnection );

        if ( connectionNode )
        {
          char*       connectionName  = xercesc::XMLString::transcode( connectionNode->getAttributes()->item( 0 )->getNodeValue() );
          std::string sConnectionName = connectionName;
          xercesc::XMLString::release( &connectionName );

          xercesc::DOMNodeList* parameterList = connectionNode->getChildNodes();

          if ( parameterList )
          {
            XMLSize_t numberOfParameters = parameterList->getLength();

            for ( XMLSize_t iParameter = 0; iParameter < numberOfParameters; ++iParameter )
            {
              xercesc::DOMNode* parameterNode = parameterList->item( iParameter );

              if ( parameterNode && parameterNode->getNodeType() == xercesc::DOMNode::ELEMENT_NODE )
              {
                char* nodeName = xercesc::XMLString::transcode( parameterNode->getNodeName() );
                std::string sNodeName = nodeName;
                xercesc::XMLString::release( &nodeName );

                if ( sNodeName == "parameter" ) { // The default parameters
                  char* parameterName = xercesc::XMLString::transcode( parameterNode->getAttributes()->item( 0 )->getNodeValue() );
                  std::string sParameterName = parameterName;
                  xercesc::XMLString::release( &parameterName );
                  char* parameterValue = xercesc::XMLString::transcode( parameterNode->getAttributes()->item( 1 )->getNodeValue() );
                  std::string sParameterValue = parameterValue;
                  xercesc::XMLString::release( &parameterValue );

		  data.registerItem( sConnectionName, sParameterName, sParameterValue );
                }
                else if ( sNodeName == "role" ) { // A role
                  char* roleName  = xercesc::XMLString::transcode( parameterNode->getAttributes()->item( 0 )->getNodeValue() );
                  std::string sRoleName = roleName;
                  xercesc::XMLString::release( &roleName );

                  // Retrieve the parameters for the role
                  xercesc::DOMNodeList* roleParameterList = parameterNode->getChildNodes();


                  if ( roleParameterList )
                  {
                    XMLSize_t numberOfRoleParameters = roleParameterList->getLength();

                    for ( XMLSize_t iRoleParameter = 0; iRoleParameter < numberOfRoleParameters; ++iRoleParameter )
                    {
                      xercesc::DOMNode* roleParameterNode = roleParameterList->item( iRoleParameter );
                      if ( roleParameterNode && roleParameterNode->getNodeType() == xercesc::DOMNode::ELEMENT_NODE )
                      {
                        char* roleNodeName = xercesc::XMLString::transcode( roleParameterNode->getNodeName() );
                        std::string sRoleNodeName = roleNodeName;
                        xercesc::XMLString::release( &roleNodeName );

                        if ( sRoleNodeName == "parameter" ) {
                          char* roleParameterName = xercesc::XMLString::transcode( roleParameterNode->getAttributes()->item( 0 )->getNodeValue() );
                          std::string sRoleParameterName = roleParameterName;
                          xercesc::XMLString::release( &roleParameterName );
                          char* roleParameterValue = xercesc::XMLString::transcode( roleParameterNode->getAttributes()->item( 1 )->getNodeValue() );
                          std::string sRoleParameterValue = roleParameterValue;
                          xercesc::XMLString::release( &roleParameterValue );

			  data.registerItem( sConnectionName, sRoleName, sRoleParameterName, sRoleParameterValue );
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
  }
  catch ( const xercesc::XMLException& toCatch )
  {
    char* message = xercesc::XMLString::transcode( toCatch.getMessage() );
    //coral::MessageStream log( serviceName );
    //log << coral::Error << message << coral::MessageStream::endmsg;
    xercesc::XMLString::release( &message );
    result = false;
  }
  catch ( const xercesc::DOMException& toCatch )
  {
    char* message = xercesc::XMLString::transcode( toCatch.msg );
    //coral::MessageStream log( serviceName );
    //log << coral::Error << message << coral::MessageStream::endmsg;
    xercesc::XMLString::release( &message );
    result = false;
  }
  catch (...)
  {
    //coral::MessageStream log( serviceName );
    //log << coral::Error << "Unexpected Exception parsing file \"" << inputFileName << "\"" << coral::MessageStream::endmsg;
    result = false;
  }

  xercesc::XMLPlatformUtils::Terminate();

  return result;
}

#include "RelationalAccess/IAuthenticationCredentials.h"
#include <iomanip>

coral_bridge::XMLAuthenticationFileContent::XMLAuthenticationFileContent(std::ostream& out) :
  m_out(out),
  m_connectionListOpen(false),
  m_connectionEntryOpen(false),
  m_roleEntryOpen(false),
  m_ind(0){
  m_out << "<?xml version=\"1.0\" ?>"<<std::endl;
  m_out << "<connectionlist>"<<std::endl;
  m_connectionListOpen = true;
}

bool coral_bridge::XMLAuthenticationFileContent::openConnectionEntry(const std::string& pcs){
  bool ret = false;
  if(m_connectionListOpen && !m_connectionEntryOpen) {
    m_out << std::endl;
    m_ind+=2;
    m_out << std::setw(m_ind)<<"";
    m_out << "<connection name=\""<<pcs<<"\" >"<<std::endl;
    m_connectionEntryOpen=true;
    ret = true;
  }
  return ret;
}

bool coral_bridge::XMLAuthenticationFileContent::closeConnectionEntry(){
  bool ret = false;
  if(m_connectionEntryOpen) {
    m_out << std::setw(m_ind)<<"";
    m_out << "</connection>"<<std::endl;
    m_ind-=2;
    ret = true;
    m_connectionEntryOpen = false;
  }
  return ret;
}

bool coral_bridge::XMLAuthenticationFileContent::openRoleEntry(const std::string& roleName){
  bool ret = false;
  if(m_connectionEntryOpen && !m_roleEntryOpen) {
    m_ind+=2;
    m_out << std::setw(m_ind)<<"";
    m_out << "<role name=\""<<roleName<<"\" >"<<std::endl;
    m_roleEntryOpen=true;
    ret = true;
  }
  return ret;

}

bool coral_bridge::XMLAuthenticationFileContent::closeRoleEntry(){
  bool ret = false;
  if(m_roleEntryOpen) {
    m_out << std::setw(m_ind)<<"";
    m_out << "</role>"<<std::endl;
    m_ind-=2;
    ret = true;
    m_roleEntryOpen = false;
  }
  return ret;
}

bool coral_bridge::XMLAuthenticationFileContent::addCredentialEntry(const std::string& userName,
                                                             const std::string& password ){
  bool ret = false;
  if(m_connectionEntryOpen) {
    m_out << std::setw(m_ind+2)<<"";
    m_out << "<parameter name=\""<<coral::IAuthenticationCredentials::userItem()<<"\" value=\""<<userName<<"\" />"<<std::endl;
    m_out << std::setw(m_ind+2)<<"";
    m_out << "<parameter name=\""<<coral::IAuthenticationCredentials::passwordItem()<<"\" value=\""<<password<<"\" />"<<std::endl;
    ret = true;
  }
  return ret;
}

void coral_bridge::XMLAuthenticationFileContent::close(){
  if(m_connectionListOpen) {
    if(m_connectionEntryOpen) {
      if(m_roleEntryOpen) {
        closeRoleEntry();
      }
      closeConnectionEntry();
    }
    m_out << std::endl;
    m_out << "</connectionlist>"<<std::endl;
  }
  m_connectionListOpen = false;
}

