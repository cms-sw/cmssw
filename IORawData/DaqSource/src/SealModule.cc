#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "IORawData/DaqSource/src/DaqSource.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(DaqSource)


#include "DaqFakeReader.h"
#include <IORawData/DaqSource/interface/DaqReaderPluginFactory.h>
DEFINE_SEAL_PLUGIN (DaqReaderPluginFactory, DaqFakeReader, "DaqFakeReader");
