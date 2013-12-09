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

class TBufferFile;

namespace cond {

  // Archives for the streaming based on ROOT.

  // output
  class RootOutputArchive {
  public:
    explicit RootOutputArchive( std::ostream& destination );

    template <typename T>
    RootOutputArchive& operator<<( const T& instance );
  private:
    // type and ptr of the object to stream
    void write( const std::type_info& sourceType, const void* sourceInstance);
  private:
    // here is where the write function will write on...
    std::ostream& m_buffer;
  };

  template <typename T> inline RootOutputArchive& RootOutputArchive::operator<<( const T& instance ){
    write( typeid(T), &instance );
    return *this;
  }

  // input
  class RootInputArchive {
  public:
    explicit RootInputArchive( std::istream& source );

    virtual ~RootInputArchive();

    template <typename T>
    RootInputArchive& operator>>( T& instance );
  private:
    // type and ptr of the object to restore
    void read( const std::type_info& destinationType, void* destinationInstance);
  private:
    // copy of the input stream. is referenced by the TBufferFile.
    std::string m_buffer;
    TBufferFile* m_streamer = nullptr;
  };

  template <typename T> inline RootInputArchive& RootInputArchive::operator>>( T& instance ){
    read( typeid(T), &instance );
    return *this;
  }

  typedef RootInputArchive CondInputArchive;
  typedef RootOutputArchive CondOutputArchive;

  template <typename T> Binary serialize( const T& payload ){
    // save data to buffer
    std::ostringstream buffer;
    CondOutputArchive oa( buffer );
    oa << payload;
    Binary ret;
    //TODO: avoid (2!!) copies
    ret.copy( buffer.str() );
    return ret;
  }

  template <typename T> boost::shared_ptr<T> deserialize( const std::string& payloadType, const Binary& payloadData){
    // for the moment we fail if types don't match... later we will check for base types...
    if( demangledName( typeid(T) )!= payloadType ) throwException(std::string("Type mismatch, target object is type \"")+payloadType+"\"",
								  "deserialize" );
    std::stringbuf sbuf;
    sbuf.pubsetbuf( static_cast<char*>(const_cast<void*>(payloadData.data())), payloadData.size() );

    std::istream buffer( &sbuf );
    CondInputArchive ia(buffer);
    boost::shared_ptr<T> payload( new T );
    ia >> (*payload);
    return payload;
  }

}
#endif
