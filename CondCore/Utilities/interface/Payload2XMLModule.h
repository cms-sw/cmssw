#include <string>
#include <memory>

#include <boost/python.hpp>
#include "boost/archive/xml_oarchive.hpp"

#include "CondFormats/Serialization/interface/Archive.h"

#define PPCAT_NX(A, B) A ## B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define STRINGIZE_NX(A) #A
#define STRINGIZE(A) STRINGIZE_NX(A)

#define PAYLOAD_2XML_MODULE( MODULE_NAME ) \
  BOOST_PYTHON_MODULE( MODULE_NAME ) 

#define PAYLOAD_2XML_CLASS( CLASS_NAME ) \
  boost::python::class_< Payload2xml<CLASS_NAME> >( STRINGIZE(PPCAT(CLASS_NAME,2xml)), boost::python::init<>()) \
  .def("write",&Payload2xml<CLASS_NAME>::write ) \
  ; 

namespace { // Avoid cluttering the global namespace.

  template <typename PayloadType> class Payload2xml {
  public:
    Payload2xml(){
    }
    //
    std::string write( const std::string &payloadData ){
      // now to convert
      std::unique_ptr< PayloadType > payload;
      std::stringbuf sdataBuf;
      sdataBuf.pubsetbuf( const_cast<char *> ( payloadData.c_str() ), payloadData.size() );

      std::istream inBuffer( &sdataBuf );
      eos::portable_iarchive ia( inBuffer );
      payload.reset( new PayloadType );
      ia >> (*payload);

      // now we have the object in memory, convert it to xml in a string and return it
      std::ostringstream outBuffer;
      {
	boost::archive::xml_oarchive xmlResult( outBuffer );
	xmlResult << boost::serialization::make_nvp( "cmsCondPayload", *payload );
      }
      return outBuffer.str();
    }
  };

} // end namespace

