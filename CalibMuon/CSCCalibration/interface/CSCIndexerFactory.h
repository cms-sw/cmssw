#ifndef CSCIndexerFactory_H
#define CSCIndexerFactory_H

#include "CalibMuon/CSCCalibration/interface/CSCIndexerBase.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<CSCIndexerBase *(void)> CSCIndexerFactory;

#endif
