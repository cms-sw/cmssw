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
//
// temporarely
#include <boost/shared_ptr.hpp>

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

  // call for the serialization. Setting packingOnly = TRUE the data will stay in the original memory layout 
  // ( no serialization in this case ). This option is used by the ORA backend - will be dropped after the changeover
  template <typename T> std::pair<Binary,Binary> serialize( const T& payload, bool packingOnly = false ){
    std::pair<Binary,Binary> ret;
    if( !packingOnly ){
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
    } else {
      // ORA objects case: nothing to serialize, the object is kept in memory in the original layout - the bare pointer is exchanged
      ret.first = Binary( payload );
    }
    return ret;
  }

  // generates an instance of T from the binary serialized data. With unpackingOnly = true the memory is already storing the object in the final 
  // format. Only a cast is required in this case - Used by the ORA backed, will be dropped in the future.
  template <typename T> boost::shared_ptr<T> default_deserialize( const std::string& payloadType, 
								  const Binary& payloadData, 
								  const Binary& streamerInfoData, 
								  bool unpackingOnly ){
    boost::shared_ptr<T> payload;
    if( !unpackingOnly ){
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
    } else {
      // ORA objects case: nothing to de-serialize, the object is already in memory in the final layout, ready to be casted
      payload = boost::static_pointer_cast<T>(payloadData.oraObject().makeShared());
    }
    return payload;
  }

  // default specialization
  template <typename T> boost::shared_ptr<T> deserialize( const std::string& payloadType, 
							  const Binary& payloadData, 
							  const Binary& streamerInfoData, 
							  bool unpackingOnly = false){
    return default_deserialize<T>( payloadType, payloadData, streamerInfoData, unpackingOnly );
 }

}

#define DESERIALIZE_BASE_CASE( BASETYPENAME )  \
  if( payloadType == #BASETYPENAME ){ \
    return default_deserialize<BASETYPENAME>( payloadType, payloadData, streamerInfoData, unpackingOnly ); \
  } 

#define DESERIALIZE_POLIMORPHIC_CASE( BASETYPENAME, DERIVEDTYPENAME )	\
  if( payloadType == #DERIVEDTYPENAME ){ \
    return boost::dynamic_pointer_cast<BASETYPENAME>( default_deserialize<DERIVEDTYPENAME>( payloadType, payloadData, streamerInfoData, unpackingOnly ) ); \
  }
 
#endif
