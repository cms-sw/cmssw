import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("writeBeamProfile2DB")

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
    tag_name = 'simBS_tag'
else:
    tag_name = options.inputTag

#################################
# Produce a SQLITE FILE
#################################
CondDBSimBeamSpotObjects = CondDB.clone(connect = cms.string('sqlite_file:test_%s.db' % tag_name)) # choose an output name
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBSimBeamSpotObjects,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("SimBeamSpotObjectsRcd"), # SimBeamSpot record
                                                                     tag = cms.string(tag_name))),                 # choose your favourite tag
                                          loadBlobStreamer = cms.untracked.bool(False)
                                          )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

from CondTools.BeamSpot.beamProfile2DBWriter_cfi import beamProfile2DBWriter
process.BeamProfile2DBWriter = beamProfile2DBWriter.clone(X0        = 0.0458532,
                                                          Y0        = -0.016966,
                                                          Z0        = -0.074992,
                                                          SigmaZ    = 3.6,
                                                          BetaStar  = 30.0,
                                                          Emittance = 3.931e-8,)

process.p = cms.Path(process.BeamProfile2DBWriter)
