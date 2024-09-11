import FWCore.ParameterSet.Config as cms
from Configuration.Geometry.defaultPhase2ConditionsEra_cff import DEFAULT_VERSION

geometry_import_stmt = f"from Configuration.Geometry.GeometryExtended{DEFAULT_VERSION}_cff import *"
exec(geometry_import_stmt)
