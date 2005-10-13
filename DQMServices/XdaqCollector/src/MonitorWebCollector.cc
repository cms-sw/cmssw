#include "Utilities/Configuration/interface/Architecture.h"
#include "SubfarmManager/MonitorWebCollector/interface/MonitorWebCollector.h"

//
// provides factory method for instantion of HellWorld application
//
HistogramServerRoot *MonitorWebCollector::DummyConsumerServer::instance_=0;

XDAQ_INSTANTIATE(MonitorWebCollector)

