#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/DataRecord/interface/RunNumberRcd.h"
#include "CondFormats/RunInfo/interface/RunNumber.h"
DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(RunNumberRcd,RunNumber);
