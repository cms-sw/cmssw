#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "DTROS8FileReader.h"
#include "DTROS25FileReader.h"
#include "DTDDUFileReader.h"
#include "DTSpyReader.h"
#include <IORawData/DaqSource/interface/DaqReaderPluginFactory.h>

DEFINE_EDM_PLUGIN (DaqReaderPluginFactory, DTROS8FileReader, "DTROS8FileReader");
DEFINE_EDM_PLUGIN (DaqReaderPluginFactory, DTROS25FileReader, "DTROS25FileReader");
DEFINE_EDM_PLUGIN (DaqReaderPluginFactory, DTDDUFileReader, "DTDDUFileReader");
DEFINE_EDM_PLUGIN (DaqReaderPluginFactory, DTSpyReader, "DTSpyReader");
