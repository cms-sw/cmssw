#include "FWCore/Utilities/interface/typelookup.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "FWCore/Framework/interface/data_default_record_trait.h"
#include <DD4hep/SpecParRegistry.h>

using namespace cms;

TYPELOOKUP_DATA_REG(cms::DDCompactView);
TYPELOOKUP_DATA_REG(DDDetector);
TYPELOOKUP_DATA_REG(dd4hep::SpecParRegistry);
TYPELOOKUP_DATA_REG(DDVectorRegistry);

EVENTSETUP_DATA_DEFAULT_RECORD(DDDetector, GeometryFileRcd);
