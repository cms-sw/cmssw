#include "FWCore/Utilities/interface/typelookup.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"

TYPELOOKUP_DATA_REG( cms::DDDetector );

#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"

#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
EVENTSETUP_RECORD_REG(DetectorDescriptionRcd);

#include "FWCore/Framework/interface/data_default_record_trait.h"
EVENTSETUP_DATA_DEFAULT_RECORD( cms::DDDetector, DetectorDescriptionRcd )
