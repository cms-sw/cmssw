

#include "CondCore/Utilities/interface/PayloadToXML.h"
#include "CondCore/Utilities/src/CondFormats.h"

namespace { // Avoid cluttering the global namespace.

  // converter methods
  std::string BeamSpotObjects2xml( std::string const &payloadData, std::string const &payloadType ) { 
    return cond::convertToXML<BeamSpotObjects> (payloadData, payloadType);
  }

} // end namespace


BOOST_PYTHON_MODULE( pluginBeamSpotObjects_toXML )
{
    using namespace boost::python;
    def ("BeamSpotObjects2xml"   , &BeamSpotObjects2xml);

}
