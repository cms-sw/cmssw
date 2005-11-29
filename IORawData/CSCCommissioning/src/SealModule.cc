#include "PluginManager/ModuleDef.h"
#include "CSCFileReader.h"
#include <IORawData/DaqSource/interface/DaqReaderPluginFactory.h>
DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN (DaqReaderPluginFactory, CSCFileReader, "CSCFileReader");
