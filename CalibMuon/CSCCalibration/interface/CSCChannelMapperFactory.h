#ifndef CSCChannelMapperFactory_H
#define CSCChannelMapperFactory_H

#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperBase.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<CSCChannelMapperBase *(void)> CSCChannelMapperFactory;

#endif
