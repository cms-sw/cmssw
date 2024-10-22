#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/PCLConfig/interface/AlignPCLThresholds.h"
#include "CondFormats/PCLConfig/interface/AlignPCLThresholdsHG.h"
#include "CondFormats/DataRecord/interface/AlignPCLThresholdsRcd.h"
#include "CondFormats/DataRecord/interface/AlignPCLThresholdsHGRcd.h"

REGISTER_PLUGIN(AlignPCLThresholdsRcd, AlignPCLThresholds);
REGISTER_PLUGIN(AlignPCLThresholdsHGRcd, AlignPCLThresholdsHG);
