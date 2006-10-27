#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "IORawData/DaqSource/src/DaqSource.h"

// The DaqSource input source
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(DaqSource);


// DaqFakeReader as a SEAL plugin
#include "DaqFakeReader.h"
#include <IORawData/DaqSource/interface/DaqReaderPluginFactory.h>
DEFINE_SEAL_PLUGIN (DaqReaderPluginFactory, DaqFakeReader, "DaqFakeReader");
