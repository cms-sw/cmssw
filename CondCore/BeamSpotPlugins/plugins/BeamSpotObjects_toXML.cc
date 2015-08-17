
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

#include "CondCore/Utilities/src/CondFormats.h"

namespace { // Avoid cluttering the global namespace.


  void initpluginBeamSpotObjects_toXML() {}

  std::string payload2xml( const std::string &payloadData, const std::string &payloadType ) { 

      // now to convert
      std::unique_ptr< BeamSpotObjects > payload;

      std::stringbuf sdataBuf;
      sdataBuf.pubsetbuf( const_cast<char *> ( payloadData.c_str() ), payloadData.size() );

      std::istream inBuffer( &sdataBuf );
      eos::portable_iarchive ia( inBuffer );
      payload.reset( new BeamSpotObjects );
      ia >> (*payload);

      // now we have the object in memory, convert it to xml in a string and return it
     
      std::ostringstream outBuffer;
      boost::archive::xml_oarchive xmlResult( outBuffer );
      xmlResult << boost::serialization::make_nvp( "cmsCondPayload", *payload );

      return outBuffer.str();
  }

} // end namespace


BOOST_PYTHON_MODULE( pluginBeamSpotObjects_toXML )
{
    using namespace boost::python;
    def ("payload2xml", payload2xml);

}
