#from __future__ import print_function
#from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.ParameterSet.Types import PSet

process = cms.Process("DQMTEST")

options = VarParsing.VarParsing('analysis')

options.register('runNumber',
                 100101,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number.")

options.register('runInputDir',
                 '/tmp',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Directory where the DQM files will appear.")

options.parseArguments()


process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )
process.source = cms.Source("DQMStreamerReader",
        runNumber = cms.untracked.uint32(options.runNumber),
        runInputDir = cms.untracked.string(options.runInputDir),
        streamLabel = cms.untracked.string('streamDQM'),
        scanOnce = cms.untracked.bool(True),
        minEventsPerLumi = cms.untracked.int32(1),
        delayMillis = cms.untracked.uint32(500),
        nextLumiTimeoutMillis = cms.untracked.int32(0),
        skipFirstLumis = cms.untracked.bool(False),
        deleteDatFiles = cms.untracked.bool(False),
        endOfRunKills  = cms.untracked.bool(False),
        inputFileTransitionsEachEvent = cms.untracked.bool(False),
        SelectEvents = cms.untracked.vstring("HLT*Mu*","HLT_*Physics*")
)

