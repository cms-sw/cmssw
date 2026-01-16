import FWCore.ParameterSet.Config as cms
from Geometry.MTDCommonData.defaultMTDConditionsEra_cff import MTD_DEFAULT_VERSION

reco_dd4hep_geometry_import_stmt = f"from Configuration.Geometry.GeometryDD4hepExtended{MTD_DEFAULT_VERSION}Reco_cff import *"
exec(reco_dd4hep_geometry_import_stmt)
