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

class RootStreamBuffer;

namespace cond {

  // default payload factory
  template <typename T> T* createPayload( const std::string& payloadTypeName ){
    std::string userTypeName = demangledName( typeid(T) );
    if( userTypeName != payloadTypeName ) 
      throwException(std::string("Type mismatch, user type: \""+userTypeName+"\", target type: \"")+payloadTypeName+"\"",
		     "createPayload" );
    return new T;
  }

  // Archives for the streaming based on ROOT.

  // output
  class RootOutputArchive {
  public:
    RootOutputArchive( std::ostream& dataDest, std::ostream& streamerInfoDest );

    template <typename T>
    RootOutputArchive& operator<<( const T& instance );
  private:
    // type and ptr of the object to stream
    void write( const std::type_info& sourceType, const void* sourceInstance);
  private:
    // here is where the write function will write on...
    std::ostream& m_dataBuffer;
    std::ostream& m_streamerInfoBuffer;
  };

  template <typename T> inline RootOutputArchive& RootOutputArchive::operator<<( const T& instance ){
    write( typeid(T), &instance );
    return *this;
  }

  // input
  class RootInputArchive {
  public:
    RootInputArchive( std::istream& binaryData, std::istream& binaryStreamerInfo );

    virtual ~RootInputArchive();

    template <typename T>
    RootInputArchive& operator>>( T& instance );
  private:
    // type and ptr of the object to restore
    void read( const std::type_info& destinationType, void* destinationInstance);
  private:
    // copy of the input stream. is referenced by the TBufferFile.
    std::string m_dataBuffer;
    std::string m_streamerInfoBuffer;
    RootStreamBuffer* m_streamer = nullptr;
  };

  template <typename T> inline RootInputArchive& RootInputArchive::operator>>( T& instance ){
    read( typeid(T), &instance );
    return *this;
  }

  typedef RootInputArchive CondInputArchive;
  typedef RootOutputArchive CondOutputArchive;

  // call for the serialization. Setting packingOnly = TRUE the data will stay in the original memory layout 
  // ( no serialization in this case ). This option is used by the ORA backend - will be dropped after the changeover
  template <typename T> std::pair<Binary,Binary> serialize( const T& payload, bool packingOnly = false ){
    std::pair<Binary,Binary> ret;
    if( !packingOnly ){
      // save data to buffers
      std::ostringstream dataBuffer;
      std::ostringstream streamerInfoBuffer;
      CondOutputArchive oa( dataBuffer, streamerInfoBuffer );
      oa << payload;
      //TODO: avoid (2!!) copies
      ret.first.copy( dataBuffer.str() );
      ret.second.copy( streamerInfoBuffer.str() );
    } else {
      // ORA objects case: nothing to serialize, the object is kept in memory in the original layout - the bare pointer is exchanged
      ret.first = Binary( payload );
    }
    return ret;
  }

  // generates an instance of T from the binary serialized data. With unpackingOnly = true the memory is already storing the object in the final 
  // format. Only a cast is required in this case - Used by the ORA backed, will be dropped in the future.
  template <typename T> boost::shared_ptr<T> deserialize( const std::string& payloadType, 
							  const Binary& payloadData, 
							  const Binary& streamerInfoData, 
							  bool unpackingOnly = false){
    // for the moment we fail if types don't match... later we will check for base types...
    boost::shared_ptr<T> payload;
    if( !unpackingOnly ){
      std::stringbuf sdataBuf;
      sdataBuf.pubsetbuf( static_cast<char*>(const_cast<void*>(payloadData.data())), payloadData.size() );
      std::stringbuf sstreamerInfoBuf;
      sstreamerInfoBuf.pubsetbuf( static_cast<char*>(const_cast<void*>(streamerInfoData.data())), streamerInfoData.size() );

      std::istream dataBuffer( &sdataBuf );
      std::istream streamerInfoBuffer( &sstreamerInfoBuf );
      CondInputArchive ia( dataBuffer, streamerInfoBuffer );
      payload.reset( createPayload<T>(payloadType) );
      ia >> (*payload);
    } else {
      // ORA objects case: nothing to de-serialize, the object is already in memory in the final layout, ready to be casted
      payload = boost::static_pointer_cast<T>(payloadData.oraObject().makeShared());
    }
    return payload;
  }

}
#endif
