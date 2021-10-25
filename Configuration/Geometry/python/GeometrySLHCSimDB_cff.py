import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
from GeometryReaders.XMLIdealGeometryESSource.cmsGeometryDB_cff import *
from Geometry.TrackerNumberingBuilder.trackerNumbering2026GeometryDB_cfi import *
from Geometry.EcalCommonData.ecalSimulationParameters_cff   import *
from Geometry.HcalCommonData.hcalSimulationParameters_cff   import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cfi        import *
from Geometry.HcalCommonData.hcalSimulationConstants_cfi    import *
from Geometry.HcalCommonData.caloSimulationParameters_cff   import *
from Geometry.MuonNumbering.muonGeometryConstants_cff       import *
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *

