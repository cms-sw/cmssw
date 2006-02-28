#include "PluginManager/ModuleDef.h"
#include <IORawData/CSCTFCommissioning/src/CSCTFFileReader.h>
#include <IORawData/DaqSource/interface/DaqReaderPluginFactory.h>
DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN (DaqReaderPluginFactory, CSCTFFileReader, "CSCTFFileReader");
