#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IORawData/RPCFileReader/interface/RPCFileReader.h"
#include "IORawData/RPCFileReader/interface/LinkDataXMLWriter.h"
#include "IORawData/RPCFileReader/interface/LinkDataXMLReader.h"
#include "IORawData/RPCFileReader/interface/RPCDigiFilter.h"
#include "IORawData/RPCFileReader/plugins/RawToXML.h"

// The RPCFileReader input source
DEFINE_FWK_INPUT_SOURCE(RPCFileReader);
DEFINE_FWK_INPUT_SOURCE(LinkDataXMLReader);
DEFINE_FWK_MODULE(LinkDataXMLWriter);
DEFINE_FWK_MODULE(RPCDigiFilter);
DEFINE_FWK_MODULE(RawToXML);
