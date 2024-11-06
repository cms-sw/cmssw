import FWCore.ParameterSet.Config as cms
from Configuration.Geometry.defaultPhase2ConditionsEra_cff import DEFAULT_VERSION

reco_dd4hep_geometry_import_stmt = f"from Configuration.Geometry.GeometryDD4hepExtended{DEFAULT_VERSION}Reco_cff import *"
exec(reco_dd4hep_geometry_import_stmt)
