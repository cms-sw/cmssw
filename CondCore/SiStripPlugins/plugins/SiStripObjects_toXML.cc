
#include "CondCore/Utilities/interface/PayloadToXML.h"
#include "CondCore/Utilities/src/CondFormats.h"

namespace { // Avoid cluttering the global namespace.

  // converter methods
  std::string SiStripLatency2xml( std::string const &payloadData, std::string const &payloadType ) { 
    return cond::convertToXML<SiStripLatency> (payloadData, payloadType);
  }

  std::string SiStripConfObject2xml( const std::string &payloadData, const std::string &payloadType ) { 
    return cond::convertToXML<SiStripConfObject> (payloadData, payloadType);
  }

} // end namespace


BOOST_PYTHON_MODULE( pluginSiStripObjects_toXML )
{
    using namespace boost::python;
    def ("SiStripLatency2xml"   , &SiStripLatency2xml);
    def ("SiStripConfObject2xml", &SiStripConfObject2xml);

}

