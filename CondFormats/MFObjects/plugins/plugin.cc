#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "CondFormats/DataRecord/interface/MFGeometryFileRcd.h"
#include "CondFormats/DataRecord/interface/MagFieldConfigRcd.h"
#include "CondFormats/MFObjects/interface/MagFieldConfig.h"

REGISTER_PLUGIN(MagFieldConfigRcd, MagFieldConfig);
REGISTER_PLUGIN(MFGeometryFileRcd, FileBlob);
