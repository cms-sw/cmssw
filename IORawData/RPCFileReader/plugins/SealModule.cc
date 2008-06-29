#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IORawData/RPCFileReader/interface/RPCFileReader.h"
#include "IORawData/RPCFileReader/interface/LinkDataXMLWriter.h"
#include "IORawData/RPCFileReader/interface/LinkDataXMLReader.h"
#include "IORawData/RPCFileReader/interface/RPCDigiFilter.h"

// The RPCFileReader input source
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(RPCFileReader);
DEFINE_ANOTHER_FWK_INPUT_SOURCE(LinkDataXMLReader);
DEFINE_ANOTHER_FWK_MODULE(LinkDataXMLWriter);
DEFINE_ANOTHER_FWK_MODULE(RPCDigiFilter);
