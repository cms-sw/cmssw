#ifndef CondCore_CondDB_Utils_h
#define CondCore_CondDB_Utils_h

#include "CondCore/CondDB/interface/Exception.h"
//
#include <string>
#include <cxxabi.h>
#include <algorithm>
#include <iostream>

namespace cond {

  namespace {

    std::string demangledName( const std::type_info& typeInfo ){
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

    std::tuple<std::string,std::string,std::string> parseConnectionString( const std::string& connectionString ){
      size_t ptr = 0;
      size_t techEnd = connectionString.find( ':' );
      if( techEnd == std::string::npos ) throwException( "Connection string "+connectionString+" is invalid format.",
							 "parseConnectionString" );
      std::string technology = connectionString.substr(ptr,techEnd);
      std::string service("");
      ptr = techEnd+1;
      if( technology != "sqlite_file" && technology != "sqlite" ){
	if( connectionString.substr( ptr,2 )!="//" ) throwException( "Connection string "+connectionString+
								     " is invalid format for technology \""+
								     technology+"\".","parseConnectionString" );
	ptr += 2;
	size_t serviceEnd = connectionString.find( '/', ptr );
	if( serviceEnd == std::string::npos ) throwException( "Connection string "+connectionString+" is invalid.",
							      "parseConnectionString" );
	service = connectionString.substr( ptr, serviceEnd-ptr );
	ptr = serviceEnd+1;
      }
      std::string schema = connectionString.substr( ptr );
      return std::make_tuple( technology, service, schema );
    }
  }

}

#endif
