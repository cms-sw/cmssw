#include "PluginManager/ModuleDef.h"
#include "DTROS8FileReader.h"
#include <IORawData/DaqSource/interface/DaqReaderPluginFactory.h>
DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN (DaqReaderPluginFactory, DTROS8FileReader, "DTROS8FileReader");
