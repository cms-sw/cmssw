#ifndef CSCChannelMapperFactory_H
#define CSCChannelMapperFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperBase.h"

typedef edmplugin::PluginFactory< CSCChannelMapperBase* (void) > CSCChannelMapperFactory;

#endif
