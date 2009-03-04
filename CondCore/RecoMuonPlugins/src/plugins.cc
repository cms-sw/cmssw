#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/RecoMuonObjects/interface/MuScleFitDBobject.h"
#include "CondFormats/DataRecord/interface/MuScleFitDBobjectRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(MuScleFitDBobjectRcd,MuScleFitDBobject);

