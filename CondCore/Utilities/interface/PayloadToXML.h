
#include <iostream>
#include <string>
#include <memory>

#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/init.hpp>
#include <boost/python/def.hpp>
#include <iostream>
#include <string>
#include <sstream>

#include "boost/archive/xml_oarchive.hpp"
#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/Serialization/interface/Archive.h"

namespace cond { 

  template<typename T>
  std::string convertToXML( const std::string &payloadData, const std::string &payloadType ) { 

      std::unique_ptr< T > payload;
      std::stringbuf sdataBuf;
      sdataBuf.pubsetbuf( const_cast<char *> ( payloadData.c_str() ), payloadData.size() );

      std::istream inBuffer( &sdataBuf );
      eos::portable_iarchive ia( inBuffer );
      payload.reset( new T );
      ia >> (*payload);

      // now we have the object in memory, convert it to xml in a string and return it
     
      std::ostringstream outBuffer;
      boost::archive::xml_oarchive xmlResult( outBuffer );
      xmlResult << boost::serialization::make_nvp( "cmsCondPayload", *payload );

      return outBuffer.str();

  }
} // end namespace cond

