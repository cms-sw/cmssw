#include "DQMServices/XdaqCollector/interface/XdaqCollector.h"

//
// provides factory method for instantion of HellWorld application
//
CollectorRoot *XdaqCollector::DummyConsumerServer::instance_=0;

XDAQ_INSTANTIATOR_IMPL(XdaqCollector);

