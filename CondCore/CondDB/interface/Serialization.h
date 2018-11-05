#ifndef CondCore_CondDB_Serialization_h
#define CondCore_CondDB_Serialization_h
//
// Package:     CondDB
// 
/**Serialization.h CondCore/CondDB/interface/Serialization.h
   Description: functions for serializing the payload objects.  
*/
//
// Author:      Giacomo Govi
// Created:     October 2013
//
//

#include "CondCore/CondDB/interface/Binary.h"
#include "CondCore/CondDB/interface/Exception.h" 
#include "CondCore/CondDB/interface/Utils.h" 
//
#include <sstream>
#include <iostream>
#include <memory>
//
// temporarely

#include "CondFormats/Serialization/interface/Archive.h"

namespace cond {

  // default payload factory
  template <typename T> T* createPayload( const std::string& payloadTypeName ){
    std::string userTypeName = demangledName( typeid(T) );
    if( userTypeName != payloadTypeName ) 
      throwException(std::string("Type mismatch, user type: \""+userTypeName+"\", target type: \"")+payloadTypeName+"\"",
		     "createPayload" );
    return new T;
  }

  template <> inline std::string* createPayload<std::string>( const std::string& payloadTypeName ){
    std::string userTypeName = demangledName( typeid(std::string) );
    if( payloadTypeName != userTypeName && payloadTypeName != "std::string" )
      throwException(std::string("Type mismatch, user type: \"std::string\", target type: \"")+payloadTypeName+"\"",
		     "createPayload" );
    return new std::string;
  }

  class StreamerInfo {
  public:
    static constexpr char const* TECH_LABEL = "technology";
    static constexpr char const* TECH_VERSION_LABEL = "tech_version";
    static constexpr char const* CMSSW_VERSION_LABEL = "CMSSW_version";
    static constexpr char const* ARCH_LABEL = "architecture";
    //
    static constexpr char const* TECHNOLOGY = "boost/serialization" ;
    static std::string techVersion();
    static std::string jsonString();
  };

  typedef cond::serialization::InputArchive  CondInputArchive;
  typedef cond::serialization::OutputArchive CondOutputArchive;

  // call for the serialization. 
  template <typename T> std::pair<Binary,Binary> serialize( const T& payload ){
    std::pair<Binary,Binary> ret;
    std::string streamerInfo( StreamerInfo::jsonString() );
    try{
      // save data to buffers
      std::ostringstream dataBuffer;
      CondOutputArchive oa( dataBuffer );
      oa << payload;
      //TODO: avoid (2!!) copies
      ret.first.copy( dataBuffer.str() );
      ret.second.copy( streamerInfo );
    } catch ( const std::exception& e ){
      std::string em( e.what() );
      throwException("Serialization failed: "+em+". Serialization info:"+streamerInfo,"serialize");
    }
    return ret;
  }

  // generates an instance of T from the binary serialized data. 
  template <typename T> std::unique_ptr<T> default_deserialize( const std::string& payloadType,
                                                                const Binary& payloadData,
                                                                const Binary& streamerInfoData ){
    std::unique_ptr<T> payload;
    std::stringbuf sstreamerInfoBuf;
    sstreamerInfoBuf.pubsetbuf( static_cast<char*>(const_cast<void*>(streamerInfoData.data())), streamerInfoData.size() );
    std::string streamerInfo = sstreamerInfoBuf.str();
    try{
      std::stringbuf sdataBuf;
      sdataBuf.pubsetbuf( static_cast<char*>(const_cast<void*>(payloadData.data())), payloadData.size() );
      std::istream dataBuffer( &sdataBuf );
      CondInputArchive ia( dataBuffer );
      payload.reset( createPayload<T>(payloadType) );
      ia >> (*payload);
    } catch ( const std::exception& e ){
      std::string errorMsg("De-serialization failed: ");
      std::string em( e.what() );
      if( em == "unsupported version" )  {
	errorMsg += "the current boost version ("+StreamerInfo::techVersion()+
	  ") is unable to read the payload. Data might have been serialized with an incompatible version.";
      } else if( em == "input stream error" ) {
	errorMsg +="data size does not fit with the current class layout. The Class "+payloadType+" might have been changed with respect to the layout used in the upload.";
      } else {
	errorMsg += em;
      }
      if( !streamerInfo.empty() ) errorMsg += " Payload serialization info: "+streamerInfo;
      throwException( errorMsg, "default_deserialize" );
    }
    return payload;
  }

  // default specialization
  template <typename T> std::unique_ptr<T> deserialize( const std::string& payloadType,
                                                        const Binary& payloadData,
                                                        const Binary& streamerInfoData ) {
    return default_deserialize<T>( payloadType, payloadData, streamerInfoData );
 }

}

#define DESERIALIZE_BASE_CASE( BASETYPENAME )  \
  if( payloadType == #BASETYPENAME ){ \
    return default_deserialize<BASETYPENAME>( payloadType, payloadData, streamerInfoData ); \
  } 

#define DESERIALIZE_POLIMORPHIC_CASE( BASETYPENAME, DERIVEDTYPENAME )	\
  if( payloadType == #DERIVEDTYPENAME ){ \
    return default_deserialize<DERIVEDTYPENAME>( payloadType, payloadData, streamerInfoData ); \
  }
 
#endif
