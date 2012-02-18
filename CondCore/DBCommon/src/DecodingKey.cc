#include "CondCore/DBCommon/interface/DecodingKey.h"
#include "CondCore/DBCommon/interface/FileUtils.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Cipher.h"
//
#include <sstream>
#include <string.h>
#include <fstream>
#include <vector>
//#include <unistd.h>
#include <pwd.h>
//#include <cstdlib>
#include <ctime>

static char ElementSeparator(',');
static char ItemSeparator(';');
static char LineSeparator('!');

// character set same as base64, except for last two (missing are + and / ) 
static const char* b64str =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

static const std::string USERPREFIX("U=");
static const std::string GROUPSPREFIX("G=");
static const std::string SERVICEPREFIX("S=");
static const std::string DATEPREFIX("D=");

static const std::string CONNECTIONPREFIX("C=");
static const std::string KEYPREFIX("K=");
static const std::string PASSWORDPREFIX("P=");

static const std::string DEFAULT_SERVICE("Cond_Default_Service");

static const size_t keySize = 10;

namespace cond {
  char randomChar(){
    int irand = ::rand()%(::strlen(b64str));
    return b64str[irand];
  }

  // the lines of the key file are prefixed by a random-sized sequence of random characters
  /**
  std::string writeLine( const std::string& encodedString ){
    ::srand( ::time( NULL)%10 );
    int irand = ::rand()%sizeof(b64str);
    std::string ret( b64str[irand] );
    for(int i=0; i<irand; i++){
      ret += randomChar();
    }
    ret += encodedString;
    return ret;
  }

  std::string readLine( const std::string& rowLine ){
    if( rowLine.empty()) return rowLine;
    size_t nskip = 0;
    for (size_t i=0;i<sizeof(b64str); i++){
      if(b64str[i]==rowLine[0]){
	nskip = i;
      }
    }
    return rowLine.sustr( nskip );
  }
  **/

  std::string getUserName(){
    std::string userName("");
    struct passwd* userp = ::getpwuid(::getuid());
    if(userp) {
      char* uName = userp->pw_name;
      if(uName){
	userName += uName;
      }
    }
    if(userName.empty()){
      std::string  msg("Cannot determine login name.");
      throwException(msg,"DecodingKey::getUserName");     
    }
    return userName;
  }

  bool validateInput(const std::string& dataSource, const std::string& key){
    if(dataSource.empty()){
      std::string msg("Provided data source connection string is empty.");
      throwException(msg,"DecodingKey::validateInput");    
    }
    if(key.find(ItemSeparator)!=std::string::npos){
      std::string msg("Invalid character ';' found in key string.");
      throwException(msg,"DecodingKey::validateInput");    
    }
    if(key.find(LineSeparator)!=std::string::npos){
      std::string msg("Invalid character '!' found in key string.");
      throwException(msg,"DecodingKey::validateInput");    
    }
    return true;
  }

}

const std::string cond::DecodingKey::FILE_NAME("cond_auth_key.dat");

std::string cond::DecodingKey::generateKey(){
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

size_t cond::DecodingKey::init( const std::string& keyFileName, const std::string& password, bool readMode ){
  if(keyFileName.empty()){
    std::string msg("Provided key file name is empty.");
    throwException(msg,"DecodingKey::init");    
  }
  m_fileName = keyFileName;
  m_pwd = password;
  m_mode = readMode;
  m_user.clear();
  m_groups.clear();
  m_serviceKeys.clear();
  size_t nelem = 0;
  if( m_mode ){
    std::ifstream keyFile (m_fileName.c_str());
    if (keyFile.is_open()){
      Cipher cipher( m_pwd );
      if ( keyFile.good() ){
	std::string encodedLine;
	getline (keyFile,encodedLine);
	std::string content = cipher.decrypt( encodedLine );
	std::stringstream str( content );
	while( str.good() ){
	  std::string line;
	  getline ( str, line,LineSeparator ); 
	  if(line.size()>3 ){
	    if( line.substr(0,2)==USERPREFIX ){
	      m_user = line.substr(2);
	    } else if ( line.substr(0,2)== GROUPSPREFIX ){
	      std::istringstream groupStr( line.substr(2) );
	      while( groupStr.good() ){
		std::string group("");
		getline( groupStr, group, ElementSeparator);
		m_groups.insert( group );
	      }
	    } else if ( line.substr(0,2)== SERVICEPREFIX ){
	      std::stringstream serviceStr( line.substr(2) );
	      std::vector<std::string> sdata;
	      while( serviceStr.good() ){
		sdata.push_back( std::string("") );
		getline( serviceStr, sdata.back(), ItemSeparator);
	      }
	      std::map< std::string, ServiceKey >::iterator iS =  m_serviceKeys.insert( std::make_pair( sdata[0], ServiceKey() ) ).first;
	      iS->second.dataSource = sdata[1];
	      iS->second.key = sdata[2];
	      iS->second.userName = sdata[3];
	      iS->second.password = sdata[4];
	      nelem++;
	    }
	  }
	}
      }
      keyFile.close();
      if( !m_user.empty() ){
	std::string currentUser = getUserName();
	if(m_user != getUserName() ){
	  m_user.clear();
	  m_groups.clear();
	  m_serviceKeys.clear();
	  std::string msg("Provided key file is invalid for user=");
	  msg+=currentUser;
	  throwException(msg,"DecodingKey::init");    
	}
      }
    } else {
      std::string msg("");
      msg += "Provided Key File \""+m_fileName+"\n is invalid.";
      throwException(msg,"DecodingKey::init");      
    }
  }
  return nelem;
}

size_t cond::DecodingKey::createFromInputFile( const std::string& inputFileName, bool genKey ){
  size_t nelem = 0;
  if(inputFileName.empty()){
    std::string msg("Provided input file name is empty.");
    throwException(msg,"DecodingKey::readFromInputFile");    
  }
  m_user.clear();
  m_groups.clear();
  m_serviceKeys.clear();
  std::ifstream inputFile (inputFileName.c_str());
  if (inputFile.is_open()){
    while ( inputFile.good() ){
      std::string line;
      getline (inputFile, line);
      if(line.size()>3 ){
	if( line.substr(0,2)==USERPREFIX ){
	  m_user = line.substr(2);
	} else if ( line.substr(0,2)== GROUPSPREFIX ){
	  std::istringstream groupStr( line.substr(2) );
	  while( groupStr.good() ){
	    std::string group("");
	    getline( groupStr, group, ElementSeparator);
	    m_groups.insert( group );
	  }
	} else if ( line.substr(0,2)== SERVICEPREFIX ){
	  std::stringstream str( line );
	  std::string service("");
	  ServiceKey skey;
	  while( str.good() ){
	    std::string keyItem;
	    getline( str, keyItem, ItemSeparator);
	    if( keyItem.size()>3 ){
	      std::string prefix = keyItem.substr(0,2);
	      std::string dt = keyItem.substr(2);
	      if( prefix==SERVICEPREFIX ){
		service = dt;
	      } else if ( prefix==CONNECTIONPREFIX ){
		skey.dataSource = dt;
	      } else if ( prefix==USERPREFIX ){
		skey.userName = dt;
	      } else if ( prefix==PASSWORDPREFIX ){
		skey.password = dt;
	      }
	    }  
	  }
	  if( genKey ) skey.key = generateKey();
	  m_serviceKeys.insert( std::make_pair( service, skey ) );
	  nelem++;
	}
      }
    }
    inputFile.close();
  } else {
    std::string msg("");
    msg += "Provided Input File \""+inputFileName+"\n is invalid.";
    throwException(msg,"DecodingKey::readFromInputFile");      
  }
  return nelem;
}

void cond::DecodingKey::list( std::ostream& out ){
  out << "## USER="<<m_user<<std::endl;
  out <<"## GROUPS=";
  bool more = false;
  for( std::set<std::string>::const_iterator ig = m_groups.begin();
       ig != m_groups.end(); ++ig ){
    if( more ) out <<",";
    out <<*ig;
    more = true;
  }
  out <<std::endl;
  for( std::map< std::string, ServiceKey >::const_iterator iS = m_serviceKeys.begin();
       iS != m_serviceKeys.end(); iS++ ){
    out <<"## SERVICE \""<<iS->first<<"\"";
    out <<" Connection="<<iS->second.dataSource<<";";
    out <<" Username="<<iS->second.userName<<";";
    out <<" Password="<<iS->second.password<<";";
    out <<" Key="<<iS->second.key<<";"<<std::endl;
  }
}

void cond::DecodingKey::flush(){
  std::ofstream outFile ( m_fileName.c_str() );
  if (outFile.is_open()){
    std::stringstream content;
    if( !m_user.empty() ){
      content << USERPREFIX << m_user << LineSeparator;
    }
    if( !m_groups.empty() ){
      content << GROUPSPREFIX;
      bool empty = true;
      for( std::set<std::string>::const_iterator iR = m_groups.begin(); iR != m_groups.end(); ++iR ){
	if( !empty ) content << ElementSeparator;
	content << *iR;
	empty = false;
      }
      content << LineSeparator;
    }
    for( std::map< std::string, ServiceKey >::const_iterator iD = m_serviceKeys.begin();
	 iD != m_serviceKeys.end(); ++iD ){
      content << SERVICEPREFIX << iD->first << ItemSeparator;
      content << iD->second.dataSource << ItemSeparator;
      content << iD->second.key << ItemSeparator;
      content << iD->second.userName << ItemSeparator;
      content << iD->second.password << ItemSeparator;
      content << LineSeparator;
    }
    Cipher cipher( m_pwd );
    outFile << cipher.encrypt( content.str() )<< std::endl;
  } else {
    std::string msg("");
    msg += "Provided Key File \""+m_fileName+"\n is invalid.";
    throwException(msg,"DecodingKey::flush");            
  }
  outFile.close();
}
  
void cond::DecodingKey::setUser( const std::string& user ){
  m_user = user;
}

void cond::DecodingKey::addGroup( const std::string& group ){
  if( !group.empty() ){
    m_groups.insert( group );
  }
}

void cond::DecodingKey::addKeyForDefaultService( const std::string& dataSource, 
						 const std::string& key ){
  addKeyForService( DEFAULT_SERVICE, dataSource,key, "", "" );
}

void cond::DecodingKey::addDefaultService( const std::string& dataSource ){
  addKeyForDefaultService( dataSource, generateKey() );  
}

void cond::DecodingKey::addKeyForService( const std::string& serviceName, 
					  const std::string& dataSource, 
					  const std::string& key, 
					  const std::string& userName, 
					  const std::string& password ){  
  validateInput( dataSource, key );
  std::map< std::string, ServiceKey >::iterator iK = m_serviceKeys.find( serviceName );
  if( iK == m_serviceKeys.end() ){
    iK = m_serviceKeys.insert( std::make_pair( serviceName, ServiceKey() ) ).first;
  }
  iK->second.dataSource = dataSource;
  iK->second.key = key;
  iK->second.userName = userName;
  iK->second.password = password;
}

void cond::DecodingKey::addService( const std::string& serviceName, 
				     const std::string& dataSource, 
				     const std::string& userName, 
				     const std::string& password ){
  addKeyForService( serviceName, dataSource, generateKey(), userName, password );
}

