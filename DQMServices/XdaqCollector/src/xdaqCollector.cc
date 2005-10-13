#include "DQMServices/xdaqCollector/interface/xdaqCollector.h"

//
// provides factory method for instantion of HellWorld application
//
CollectorRoot *xdaqCollector::DummyConsumerServer::instance_=0;

XDAQ_INSTANTIATE(xdaqCollector)

