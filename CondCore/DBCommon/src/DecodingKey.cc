#include "CondCore/DBCommon/interface/DecodingKey.h"
#include "CondCore/DBCommon/interface/FileUtils.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Cipher.h"
//
#include <sstream>
#include <string.h>
#include <fstream>
#include <vector>
#include <pwd.h>
#include <ctime>

constexpr char ItemSeparator = ';';
constexpr char LineSeparator = '!';

// character set same as base64, except for last two (missing are + and / ) 
static const char* b64str =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

static const std::string KEY_HEADER("Cond_Authentication_Key");

static const std::string NAMEPREFIX("N=");
static const std::string KEYPREFIX("K=");
static const std::string OWNERPREFIX("O=");

static const std::string DATEPREFIX("D=");

static const std::string SERVICEPREFIX("S=");
static const std::string CONNECTIONPREFIX("C=");
static const std::string USERPREFIX("U=");
static const std::string PASSWORDPREFIX("P=");

static const std::string DEFAULT_SERVICE("Cond_Default_Service");

namespace cond {
  char randomChar(){
    int irand = ::rand()%(::strlen(b64str));
    return b64str[irand];
  }

  std::string getLoginName(){
    std::string loginName("");
    struct passwd* userp = ::getpwuid(::getuid());
    if(userp) {
      char* uName = userp->pw_name;
      if(uName){
	loginName += uName;
      }
    }
    if(loginName.empty()){
      std::string  msg("Cannot determine login name.");
      throwException(msg,"DecodingKey::getLoginName");     
    }
    return loginName;
  }

  void parseLineForNamedParams( const std::string& line, std::map<std::string,std::string>& params ){
    std::stringstream str( line );
    std::string paramName("");
    std::string paramValue("");  
    while( str.good() ){
      std::string item("");
      getline( str, item, ItemSeparator);
      if( item.size()>3 ){
	paramName = item.substr(0,2);
	paramValue = item.substr(2);
	params.insert( std::make_pair( paramName, paramValue ) );
      }  
    }
  }

}

std::string cond::KeyGenerator::make( size_t keySize ){
  ::srand( m_iteration+2 );
  int rseed = ::rand();
  int seed = ::time( NULL)%10 + rseed;
  ::srand( seed );
  std::string ret("");
  for( size_t i=0;i<keySize; i++ ){
    ret += randomChar();
  }
  m_iteration++;
  return ret;
}

std::string cond::KeyGenerator::makeWithRandomSize( size_t maxSize ){
  ::srand( m_iteration+2 );
  int rseed = ::rand();
  int seed = ::time( NULL)%10 + rseed;
  ::srand( seed );
  size_t sz = rand()%maxSize;
  return make( sz );
}

const std::string cond::DecodingKey::FILE_NAME("db.key");
const std::string cond::DecodingKey::FILE_PATH(".cms_cond/"+FILE_NAME);

std::string cond::DecodingKey::templateFile(){
  std::stringstream s;
  s<<NAMEPREFIX<<"<principal_name>"<<std::endl;
  s<<OWNERPREFIX<<"<owner_name, optional>"<<std::endl;
  s<<KEYPREFIX<<"<key, leave empty if generated>"<<std::endl;
  //s<<DATEPREFIX<<"<expiring date, optional>"<<std::endl;
  s<<SERVICEPREFIX<<"<service_name0>;"<<CONNECTIONPREFIX<<"<service0_connection_string>;"<<USERPREFIX<<"<user0_name>;"<<PASSWORDPREFIX<<"<password0>;"<<std::endl;
  s<<SERVICEPREFIX<<"<service_name1>;"<<CONNECTIONPREFIX<<"<service1_connection_string>;"<<USERPREFIX<<"<user1_name>;"<<PASSWORDPREFIX<<"<password1>;"<<std::endl;
  s<<SERVICEPREFIX<<"<service_name2>;"<<CONNECTIONPREFIX<<"<service2_connection_string>;"<<USERPREFIX<<"<user2_name>;"<<PASSWORDPREFIX<<"<password2>;"<<std::endl;
  return s.str();
}

size_t cond::DecodingKey::init( const std::string& keyFileName, const std::string& password, bool readMode ){
  if(keyFileName.empty()){
    std::string msg("Provided key file name is empty.");
    throwException(msg,"DecodingKey::init");    
  }
  m_fileName = keyFileName;
  m_pwd = password;
  m_mode = readMode;
  m_principalName.clear();
  m_principalKey.clear();
  m_owner.clear();
  m_services.clear();
  size_t nelem = 0;
  if( m_mode ){
    std::ifstream keyFile (m_fileName.c_str(),std::ios::in|std::ios::binary|std::ios::ate);
    if (keyFile.is_open()){
      size_t fsize = keyFile.tellg();
      unsigned char* buff = (unsigned char*)malloc( fsize );
      keyFile.seekg (0, std::ios::beg);
      keyFile.read (reinterpret_cast<char*>(buff), fsize);
      Cipher cipher( m_pwd );
      std::string content = cipher.decrypt( buff, fsize );
      free ( buff );
      // skip the header + line separator
      if( content.substr( 0, KEY_HEADER.size() )!=KEY_HEADER ){
	std::string msg("Provided key content is invalid.");	
	throwException(msg,"DecodingKey::init");    	
      } 
      std::stringstream str( content.substr( KEY_HEADER.size()+1) );
      while( str.good() ){
	std::string line;
	getline ( str, line,LineSeparator ); 
	if(line.size()>3 ){
	  if( line.substr(0,2)==NAMEPREFIX ){
	    m_principalName = line.substr(2);
	  } else if ( line.substr(0,2)== KEYPREFIX ){
	    m_principalKey = line.substr(2);
	  } else if ( line.substr(0,2)== OWNERPREFIX ){
	    m_owner = line.substr(2);
	  } else if ( line.substr(0,2)== SERVICEPREFIX ){
	    std::stringstream serviceStr( line.substr(2) );
	    std::vector<std::string> sdata;
	    while( serviceStr.good() ){
	      sdata.push_back( std::string("") );
	      getline( serviceStr, sdata.back(), ItemSeparator);
	    }
	    std::map< std::string, ServiceCredentials >::iterator iS =  m_services.insert( std::make_pair( sdata[0], ServiceCredentials() ) ).first;
	    iS->second.connectionString = sdata[1];
	    iS->second.userName = sdata[2];
	    iS->second.password = sdata[3];
	    nelem++;
	  }
	}
      }
      keyFile.close();
      if( m_principalName.empty() || m_principalKey.empty() ){
	std::string msg = "Provided key is invalid.";
	throwException(msg,"DecodingKey::init");    
      }
      if( !m_owner.empty() ){
	std::string currentUser = getLoginName();
	if(m_owner != currentUser ){
	  m_principalName.clear();
	  m_principalKey.clear();
	  m_owner.clear();
	  m_services.clear();
	  std::string msg = "Provided key is invalid for user=" + currentUser;
	  throwException(msg,"DecodingKey::init");    
	}
      }
    } else {
      std::string msg = "Required Key File \""+m_fileName+"\" is missing or unreadable.";
      throwException(msg,"DecodingKey::init");      
    }
  }
  return nelem;
}

size_t cond::DecodingKey::createFromInputFile( const std::string& inputFileName, size_t generatedKeySize ){
  size_t nelem = 0;
  if(inputFileName.empty()){
    std::string msg("Provided input file name is empty.");
    throwException(msg,"DecodingKey::readFromInputFile");    
  }
  m_principalName.clear();
  m_principalKey.clear();
  m_owner.clear();
  m_services.clear();
  std::ifstream inputFile (inputFileName.c_str());
  if (inputFile.is_open()){
    std::map<std::string,std::string> params;
    while ( inputFile.good() ){
      std::string line;
      getline (inputFile, line);
      params.clear();
      if(line.size()>3 ){
	if( line.substr(0,2)==NAMEPREFIX ){
	  m_principalName = line.substr(2);
	} else if ( line.substr(0,2)== KEYPREFIX ){
	  m_principalKey = line.substr(2);
	} else if ( line.substr(0,2)== OWNERPREFIX ){
	  m_owner = line.substr(2);
	} else if ( line.substr(0,2)== SERVICEPREFIX ){
	  parseLineForNamedParams( line, params );
	  std::string& serviceName = params[ SERVICEPREFIX ];
	  ServiceCredentials creds;
	  creds.connectionString = params[ CONNECTIONPREFIX ];
	  creds.userName = params[ USERPREFIX ];
	  creds.password = params[ PASSWORDPREFIX ];
	  m_services.insert( std::make_pair( serviceName, creds ) );
	  nelem++;
	}
      }
    }
    inputFile.close();
    if( m_principalKey.empty() && generatedKeySize){
      KeyGenerator gen;
      m_principalKey = gen.make( generatedKeySize );
    }

  } else {
    std::string msg = "Provided Input File \""+inputFileName+"\n is invalid.";
    throwException(msg,"DecodingKey::readFromInputFile");      
  }
  return nelem;
}

void cond::DecodingKey::list( std::ostream& out ){
  out <<NAMEPREFIX<<m_principalName<<std::endl;
  out <<KEYPREFIX<<m_principalKey<<std::endl;
  out <<OWNERPREFIX<<m_owner<<std::endl;
  for( std::map< std::string, ServiceCredentials >::const_iterator iS = m_services.begin();
       iS != m_services.end(); iS++ ){
    out <<SERVICEPREFIX<<iS->first<<";";
    out <<CONNECTIONPREFIX<<iS->second.connectionString<<";";
    out <<USERPREFIX<<iS->second.userName<<";";
    out <<PASSWORDPREFIX<<iS->second.password<<";"<<std::endl;
  }
}

void cond::DecodingKey::flush(){
  std::ofstream outFile ( m_fileName.c_str(),std::ios::binary);
  if (outFile.is_open()){
    std::stringstream content;
    content << KEY_HEADER << LineSeparator;
    if( !m_principalName.empty() ){
      content << NAMEPREFIX << m_principalName << LineSeparator;
    }
    if( !m_principalKey.empty() ){
      content << KEYPREFIX << m_principalKey << LineSeparator;
    }
    if( !m_owner.empty() ){
      content << OWNERPREFIX << m_owner << LineSeparator;
    }
    for( std::map< std::string, ServiceCredentials >::const_iterator iD = m_services.begin();
	 iD != m_services.end(); ++iD ){
      content << SERVICEPREFIX << iD->first << ItemSeparator;
      content << iD->second.connectionString << ItemSeparator;
      content << iD->second.userName << ItemSeparator;
      content << iD->second.password << ItemSeparator;
      content << LineSeparator;
    }
    Cipher cipher( m_pwd );
    unsigned char* out;
    size_t outSize = cipher.encrypt( content.str(), out );
    outFile.write( reinterpret_cast<char*>(out),outSize);
    free (out );
  } else {
    std::string msg("");
    msg += "Provided Key File \""+m_fileName+"\n is invalid.";
    throwException(msg,"DecodingKey::flush");            
  }
  outFile.close();
}
  
void cond::DecodingKey::addDefaultService( const std::string& connectionString ){
  addService( DEFAULT_SERVICE, connectionString, "", "" );  
}

void cond::DecodingKey::addService( const std::string& serviceName, 
				    const std::string& connectionString, 
				    const std::string& userName, 
				    const std::string& password ){  
  std::map< std::string, ServiceCredentials >::iterator iK = m_services.find( serviceName );
  if( iK == m_services.end() ){
    iK = m_services.insert( std::make_pair( serviceName, ServiceCredentials() ) ).first;
  }
  iK->second.connectionString = connectionString;
  iK->second.userName = userName;
  iK->second.password = password;
}

