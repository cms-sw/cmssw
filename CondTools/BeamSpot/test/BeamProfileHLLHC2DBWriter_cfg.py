import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("writeBeamProfileHLLHC2DB")

options = VarParsing.VarParsing()
options.register('unitTest',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "are we running the unit test?")
options.register('inputTag',
                 "myTagName", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "output tag name")
options.parseArguments()


process.load("FWCore.MessageLogger.MessageLogger_cfi")
from CondCore.CondDB.CondDB_cfi import *

if options.unitTest :
    tag_name = 'simHLLHCBS_tag'
else:
    tag_name = options.inputTag

#################################
# Produce a SQLITE FILE
#################################
CondDBSimBeamSpotObjects = CondDB.clone(connect = cms.string('sqlite_file:test_%s.db' % tag_name)) # choose an output name
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBSimBeamSpotObjects,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("SimBeamSpotHLLHCObjectsRcd"), # SimBeamSpotHLLHCObjects record
                                                                     tag = cms.string(tag_name))),                      # choose your favourite tag
                                          loadBlobStreamer = cms.untracked.bool(False)
                                          )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

from CondTools.BeamSpot.beamProfileHLLHC2DBWriter_cfi import beamProfileHLLHC2DBWriter
process.BeamProfileHLLHC2DBWriter = beamProfileHLLHC2DBWriter.clone(EProton = 6500,
                                                                    CrabFrequency = 400,
                                                                    RF800 = False,
                                                                    CrossingAngle = 510,
                                                                    CrabbingAngleCrossing = 380.0,
                                                                    BetaCrossingPlane = 0.20,
                                                                    BetaSeparationPlane = 0.20,
                                                                    HorizontalEmittance = 2.5e-06,
                                                                    VerticalEmittance = 2.05e-06,
                                                                    BunchLength = 0.090,)

process.p = cms.Path(process.BeamProfileHLLHC2DBWriter)
