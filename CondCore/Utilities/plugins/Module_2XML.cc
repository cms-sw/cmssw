#include "CondCore/Utilities/interface/Payload2XMLModule.h"
#include "CondCore/Utilities/src/CondFormats.h"

PAYLOAD_2XML_MODULE( pluginUtilities_payload2xml ){
  PAYLOAD_2XML_CLASS( BeamSpotObjects );
  PAYLOAD_2XML_CLASS( EcalCondObjectContainer<EcalPedestal> );
  PAYLOAD_2XML_CLASS( EcalLaserAPDPNRatios );
  PAYLOAD_2XML_CLASS( RunInfo );
  PAYLOAD_2XML_CLASS( SiStripLatency );
  PAYLOAD_2XML_CLASS( SiStripConfObject );
}
