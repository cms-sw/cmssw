#ifndef CSCIndexerFactory_H
#define CSCIndexerFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerBase.h"

typedef edmplugin::PluginFactory< CSCIndexerBase* (void) > CSCIndexerFactory;

#endif
