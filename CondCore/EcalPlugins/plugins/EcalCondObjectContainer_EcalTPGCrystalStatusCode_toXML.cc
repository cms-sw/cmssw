
#include "CondCore/Utilities/interface/PayloadToXML.h"
#include "CondCore/Utilities/src/CondFormats.h"

namespace { // Avoid cluttering the global namespace.

  // converter methods
  std::string EcalCondObjectContainer_EcalTPGCrystalStatusCode2xml( std::string const &payloadData, std::string const &payloadType ) { 
    return cond::convertToXML<EcalCondObjectContainer<EcalTPGCrystalStatusCode> > (payloadData, payloadType);
  }

} // end namespace


BOOST_PYTHON_MODULE( pluginEcalCondObjectContainer_EcalTPGCrystalStatusCode_toXML )
{
  using namespace boost::python;
  def ("EcalCondObjectContainer_EcalTPGCrystalStatusCode2xml"   , &EcalCondObjectContainer_EcalTPGCrystalStatusCode2xml);

}
