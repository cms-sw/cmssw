# Produce pat::Tau collection with the new DNN Tau-Ids from miniAOD 12Apr2018_94X_mc2017

import FWCore.ParameterSet.Config as cms

# Options
#from FWCore.ParameterSet.VarParsing import VarParsing
# options = VarParsing('analysis')
# options.parseArguments()
updatedTauName = "slimmedTausNewID"
minimalOutput = True
eventsToProcess = 1000
nThreads = 2

process = cms.Process('TauID')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.GlobalTag.globaltag = '94X_mc2017_realistic_v14'

# Input source
process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring(
    # File from dataset /GluGluHToTauTau_M125_13TeV_powheg_pythia8/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM
    '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/90000/0498CD6A-CC42-E811-95D3-008CFA1CB8A8.root'
))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(eventsToProcess) )

# Add new TauIDs
import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
tauIdEmbedder = tauIdConfig.TauIDEmbedder(process, cms, debug = False,
                    updatedTauName = updatedTauName,
                    toKeep = [ "2017v2", "dR0p32017v2", "newDM2017v2",
                               "deepTau2017v1",
                               "DPFTau_2016_v0","DPFTau_2016_v1"])
tauIdEmbedder.runTauID()

# Output definition
process.out = cms.OutputModule("PoolOutputModule",
     fileName = cms.untracked.string('patTuple_newTauIDs.root'),
     compressionAlgorithm = cms.untracked.string('LZMA'),
     compressionLevel = cms.untracked.int32(4),
     outputCommands = cms.untracked.vstring('drop *')
)
if not minimalOutput:
     print("Store full MiniAOD EventContent")
     from Configuration.EventContent.EventContent_cff import MINIAODSIMEventContent
     from PhysicsTools.PatAlgos.slimming.MicroEventContent_cff import MiniAODOverrideBranchesSplitLevel
     process.out.outputCommands = MINIAODSIMEventContent.outputCommands
     process.out.overrideBranchesSplitLevel = MiniAODOverrideBranchesSplitLevel
process.out.outputCommands.append("keep *_"+updatedTauName+"_*_*")

# Path and EndPath definitions
process.p = cms.Path(
    process.rerunMvaIsolationSequence *
    getattr(process,updatedTauName)
)
process.endjob = cms.EndPath(process.endOfProcess)
process.outpath = cms.EndPath(process.out)
# Schedule definition
process.schedule = cms.Schedule(process.p,process.endjob,process.outpath)

##
process.load('FWCore.MessageLogger.MessageLogger_cfi')
if process.maxEvents.input.value()>10:
     process.MessageLogger.cerr.FwkReport.reportEvery = process.maxEvents.input.value()//10
if process.maxEvents.input.value()>10000 or process.maxEvents.input.value()<0:
     process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.options = cms.untracked.PSet(
     wantSummary = cms.untracked.bool(False),
     numberOfThreads = cms.untracked.uint32(nThreads),
     numberOfStreams = cms.untracked.uint32(0)
)
