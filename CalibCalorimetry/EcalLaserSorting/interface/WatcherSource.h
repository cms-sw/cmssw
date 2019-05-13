#ifndef WatcherSourceModule_H
#define WatcherSourceModule_H

#include "CalibCalorimetry/EcalLaserSorting/interface/WatcherStreamFileReader.h"
#include "IOPool/Streamer/interface/StreamerInputModule.h"

typedef edm::StreamerInputModule<WatcherStreamFileReader> WatcherSource;

#endif  //WatcherSourceModule_H not defined
