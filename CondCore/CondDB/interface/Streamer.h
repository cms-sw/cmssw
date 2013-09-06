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
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/shared_ptr.hpp>
//#include <boost/archive/text_oarchive.hpp>
//#include <boost/archive/text_iarchive.hpp>
//

class TBufferFile;

namespace conddb {

  // Archives for the streaming based on ROOT.

  // output
  class RootOutputArchive {
  public:
    explicit RootOutputArchive( boost::iostreams::filtering_streambuf<boost::iostreams::output>& outputData );

    template <typename T>
    RootOutputArchive& operator<<( const T& instance );
  private:
    // type and ptr of the object to stream
    void write( const std::type_info& sourceType, const void* sourceInstance);
  private:
    // here is where the write function will write on...
    boost::iostreams::filtering_streambuf<boost::iostreams::output>& m_buffer;
  };

  template <typename T> inline RootOutputArchive& RootOutputArchive::operator<<( const T& instance ){
    write( typeid(T), &instance );
    return *this;
  }

  // input
  class RootInputArchive {
  public:
    explicit RootInputArchive( boost::iostreams::filtering_streambuf<boost::iostreams::input>& inputData );
    ~RootInputArchive();

    template <typename T>
    RootInputArchive& operator>>( T& instance );
  private:
    // type and ptr of the object to restore
    void read( const std::type_info& destinationType, void* destinationInstance);
  private:
    // copy of the input stream. is referenced by the TBufferFile.
    std::string m_copy;
    TBufferFile* m_buffer = nullptr;
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
    std::ostringstream outBuf;
    outBuf.precision( 20 );
    { 
      boost::iostreams::filtering_streambuf<boost::iostreams::output> f;
      f.push(boost::iostreams::gzip_compressor());
      f.push(outBuf);
      //boost::archive::text_oarchive oa(outBuf);
      RootOutputArchive oa(f);
      oa << payload;
    } // gzip_compressor flushes when f goes out of scope
    m_data.copy( outBuf.str() );
  }

  class InputStreamer {
  public:
    InputStreamer( const std::string& payloadType, const Binary& payloadData );

    template <typename T> boost::shared_ptr<T> read();

  private:
    std::string m_objectType;
    std::ostringstream m_outBuf;
  }; 

  template <typename T> inline boost::shared_ptr<T> InputStreamer::read(){
    // for the moment we fail if types don't match... later we will check for base types...
    if( demangledName( typeid(T) )!= m_objectType ) throwException(std::string("Type mismatch, target object is type \"")+m_objectType+"\"",
								   "OutputStreamer::read" );
    boost::shared_ptr<T> payload( new T );
    {
      std::istringstream iss( m_outBuf.str() ); 
      boost::iostreams::filtering_streambuf<boost::iostreams::input> f;
      f.push(boost::iostreams::gzip_decompressor());
      f.push(iss);
      //boost::archive::text_iarchive ia(iss);
      RootInputArchive ia(f);
      ia >> (*payload);
    }
    return payload;
  }

}
#endif
