import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

# Cmd line options
options = VarParsing ('analysis')
options.register('produceByRun',
                 True,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 "Accumulate EcalPhiSym RecHits by Run or by LuminosityBlock")
options.register('saveFlatNano',
                 True,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 "Produce FlatNanoAOD output")
options.register('saveEDMNano',
                 False,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 "Produce EDMNanoAOD output")
options.register('saveEDM',
                 False,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 "Produce stadard EDM output")
options.inputFiles=[""]
options.parseArguments()

process = cms.Process("ECALPHISYM")
process.load("FWCore.MessageService.MessageLogger_cfi")

# Multi-threading
process.options.numberOfStreams = cms.untracked.uint32(4)
process.options.numberOfThreads = cms.untracked.uint32(4)

# Conditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.Geometry.GeometryExtended2018Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2018Reco_cff')

# use auto:run2_data to be able to test on run2 data for the time being.
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

# Source data
# skip bad events
process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound'),
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(options.inputFiles)
                        )

# Load EcalPhiSym workflow
from Calibration.EcalCalibAlgos.EcalPhiSymRecoSequence_cff import ecal_phisym_workflow
ecal_phisym_workflow(process, 
                     produce_by_run=options.produceByRun,
                     save_flatnano=options.saveFlatNano,
                     save_edmnano=options.saveEDMNano,
                     save_edm=options.saveEDM)

process.MessageLogger.cout.FwkSummary = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100000))
process.MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100000))
process.MessageLogger.cerr.FwkSummary.reportEvery = 100000
process.MessageLogger.cerr.FwkReport.reportEvery = 100000
