#include "CalibMuon/CSCCalibration/interface/CSCIndexerFactory.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerStartup.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerPostls1.h"

#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperFactory.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperStartup.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperPostls1.h"


DEFINE_EDM_PLUGIN(CSCIndexerFactory, CSCIndexerStartup, "CSCIndexerStartup");
DEFINE_EDM_PLUGIN(CSCIndexerFactory, CSCIndexerPostls1, "CSCIndexerPostls1");

DEFINE_EDM_PLUGIN(CSCChannelMapperFactory, CSCChannelMapperStartup, "CSCChannelMapperStartup");
DEFINE_EDM_PLUGIN(CSCChannelMapperFactory, CSCChannelMapperPostls1, "CSCChannelMapperPostls1");
