#include "FWCore/Utilities/interface/typelookup.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"
#include "DetectorDescription/DDCMS/interface/MuonNumbering.h"
#include "DetectorDescription/DDCMS/interface/MuonNumberingRcd.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
#include "FWCore/Framework/interface/data_default_record_trait.h"

using namespace cms;

TYPELOOKUP_DATA_REG(DDDetector);
TYPELOOKUP_DATA_REG(DDSpecParRegistry);
TYPELOOKUP_DATA_REG(DDVectorRegistry);
TYPELOOKUP_DATA_REG(MuonNumbering);

EVENTSETUP_RECORD_REG(DetectorDescriptionRcd);
EVENTSETUP_DATA_DEFAULT_RECORD(DDDetector, DetectorDescriptionRcd);
EVENTSETUP_RECORD_REG(DDSpecParRegistryRcd);
EVENTSETUP_RECORD_REG(DDVectorRegistryRcd);
EVENTSETUP_RECORD_REG(MuonNumberingRcd);
