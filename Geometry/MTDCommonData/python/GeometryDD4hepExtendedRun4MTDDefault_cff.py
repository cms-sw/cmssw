import FWCore.ParameterSet.Config as cms
from Geometry.MTDCommonData.defaultMTDConditionsEra_cff import MTD_DEFAULT_VERSION

dd4hep_geometry_import_stmt = f"from Configuration.Geometry.GeometryDD4hepExtended{MTD_DEFAULT_VERSION}_cff import *"
exec(dd4hep_geometry_import_stmt)
