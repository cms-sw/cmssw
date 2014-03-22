#ifndef CondCore_CondDB_Utils_h
#define CondCore_CondDB_Utils_h

#include "CondCore/CondDB/interface/Exception.h"
//
#include <string>
#include <cxxabi.h>
#include <algorithm>
#include <iostream>
#include <tuple>

namespace cond {

  namespace {

    inline std::string demangledName( const std::type_info& typeInfo ){
      int status = 0;
      std::string ret("");
      char* realname = abi::__cxa_demangle( typeInfo.name(), 0, 0, &status);
      if( status == 0 && realname ){
	ret  = realname;
	free(realname);
      }
      // clean up the spaces... ( to be removed after getting rid of reflex - that does not have spaces...)
      ret.erase( std::remove( ret.begin(), ret.end(), ' ' ), ret.end() ); 
      return ret;
    }

  }

  namespace persistency {

    inline std::string getConnectionProtocol( const std::string& connectionString ){
      size_t techEnd = connectionString.find( ':' );
      if( techEnd == std::string::npos ) throwException( "Could not resolve the connection protocol on "+connectionString+".",
							 "getConnectionProtocol" );
      std::string technology = connectionString.substr(0,techEnd);
      return technology;
    }
    
    inline std::tuple<std::string,std::string,std::string> parseConnectionString( const std::string& connectionString ){
      std::string protocol = getConnectionProtocol( connectionString );
      std::string serviceName("");
      std::string databaseName("");
      if( protocol == "sqlite" || protocol == "sqlite_file" || protocol == "sqlite_fip" ){
	databaseName = connectionString.substr( protocol.size()+1 ); 
      } else if ( protocol == "oracle" || protocol == "frontier" ){
	size_t ptr = protocol.size()+1;
	if( connectionString.substr( ptr,2 )!="//" ) throwException( "Connection string "+connectionString+
								     " is invalid format for technology \""+
								     protocol+"\".","parseConnectionString" );
	ptr += 2;
	size_t serviceEnd = connectionString.find( '/', ptr );
	if( serviceEnd == std::string::npos ) throwException( "Connection string "+connectionString+" is invalid.",
							      "parseConnectionString" );
	serviceName = connectionString.substr( ptr, serviceEnd-ptr );
	ptr = serviceEnd+1;
	databaseName = connectionString.substr( ptr );
      } else throwException( "Technology "+protocol+" is not known.","parseConnectionString" );
	
      return std::make_tuple( protocol, serviceName, databaseName );
    }
    
  }

}

#endif
