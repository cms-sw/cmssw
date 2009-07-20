#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/DQMObjects/interface/DQMSummary.h"
#include "CondFormats/DataRecord/interface/DQMSummaryRcd.h"
DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(DQMSummaryRcd, DQMSummary);
#include "CondFormats/DQMObjects/interface/HDQMSummary.h"
#include "CondFormats/DataRecord/interface/HDQMSummaryRcd.h"
REGISTER_PLUGIN(HDQMSummaryRcd, HDQMSummary);
#include "CondFormats/GeometryObjects/interface/GeometryFile.h"
#include "CondFormats/DataRecord/interface/DQMReferenceHistogramRootFileRcd.h"
REGISTER_PLUGIN(DQMReferenceHistogramRootFileRcd, GeometryFile);
