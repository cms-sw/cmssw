#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/RecoMuonObjects/interface/MuScleFitDBobject.h"
#include "CondFormats/DataRecord/interface/MuScleFitDBobjectRcd.h"
REGISTER_PLUGIN(MuScleFitDBobjectRcd, MuScleFitDBobject);

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/RecoMuonObjects/interface/DYTThrObject.h"
#include "CondFormats/DataRecord/interface/DYTThrObjectRcd.h"
REGISTER_PLUGIN(DYTThrObjectRcd, DYTThrObject);

#include "CondFormats/RecoMuonObjects/interface/MuonSystemAging.h"
#include "CondFormats/DataRecord/interface/MuonSystemAgingRcd.h"
REGISTER_PLUGIN(MuonSystemAgingRcd, MuonSystemAging);
