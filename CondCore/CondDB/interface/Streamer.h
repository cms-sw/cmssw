#ifndef CondCore_CondDB_Streamer_h
#define CondCore_CondDB_Streamer_h
//
// Package:     CondDB
// Class  :     Streamer
// 
/**\class Streamer Streamer.h CondCore/CondDB/interface/Streamer.h
   Description: functions for streaming the payload objects.  
*/
//
// Author:      Giacomo Govi
// Created:     May 2013
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

namespace conddb {

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
    explicit RootInputArchive( const std::stringbuf& source );

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

  // Generic streaming classes. Currently based on root. Could be a template class?

  class OutputStreamer {
  public:
    OutputStreamer();

    template <typename T> void write( const T& payload );

    const Binary& data() const;
  private:
    Binary m_data;
  };

  template <typename T> inline void OutputStreamer::write( const T& payload ){
    // save data to buffer
    std::stringbuf buffer;
    { 
      std::ostream s(&buffer);
      RootOutputArchive oa(s);
      oa << payload;
    } 
    m_data.copy( buffer.str() );
  }

  class InputStreamer {
  public:
    InputStreamer( const std::string& payloadType, const Binary& payloadData );

    template <typename T> boost::shared_ptr<T> read();

  private:
    std::string m_objectType;
    std::stringbuf  m_buffer;
  }; 

  template <typename T> inline boost::shared_ptr<T> InputStreamer::read(){
    // for the moment we fail if types don't match... later we will check for base types...
    if( demangledName( typeid(T) )!= m_objectType ) throwException(std::string("Type mismatch, target object is type \"")+m_objectType+"\"",
								   "OutputStreamer::read" );
    boost::shared_ptr<T> payload( new T );
    {
      RootInputArchive ia(m_buffer);
      ia >> (*payload);
    }
    return payload;
  }

}
#endif
