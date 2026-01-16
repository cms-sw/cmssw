import FWCore.ParameterSet.Config as cms
from Geometry.MTDCommonData.defaultMTDConditionsEra_cff import MTD_DEFAULT_VERSION

geometry_import_stmt = f"from Configuration.Geometry.GeometryExtended{MTD_DEFAULT_VERSION}_cff import *"
exec(geometry_import_stmt)
