###############################################################################
# Way to use this:
#   cmsRun testHGCalCellSizeHex_cfg.py geometry=D127
#
#   The default geometry is the default Run4 geometry
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings

options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D121",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D104, D110, D111, D112, D113, D114, D115, D120, D121, D122, D123, D124, D125, D126, D127, D128, D129, D130")
options.parseArguments()

geomName = "Run4" + options.geometry
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)

process = cms.Process("TestHGCalCellSizeHex", ERA)
process.load("Configuration.Geometry.GeometryExtended" + geomName + "_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.HGCalGeom = dict()

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1

process.load("Geometry.HGCalCommonData.hgcalCellSizeHexTester_cfi")
process.p1 = cms.Path(process.hgcalCellSizeHexTester)
