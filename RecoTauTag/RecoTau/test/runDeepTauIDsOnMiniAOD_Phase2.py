# Produce pat::Tau collection with the new DNN Tau-Ids from miniAOD 12Apr2018_94X_mc2017

import FWCore.ParameterSet.Config as cms

# Options
#from FWCore.ParameterSet.VarParsing import VarParsing
# options = VarParsing('analysis')
# options.parseArguments()
updatedTauName = "slimmedTausNewID"
minimalOutput = True
eventsToProcess = 100
nThreads = 1

process = cms.Process('TauID')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')

# Input source
process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring(
    # File from dataset VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5
    '/store/mc/Phase2HLTTDRWinter20RECOMiniAOD/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5/MINIAODSIM/PU200_110X_mcRun4_realistic_v3-v3/20000/1EF484CA-52F4-F044-B0CC-D4C636C5F0B9.root'
))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(eventsToProcess) )

# Add new TauIDs
import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
toKeep = [ "2017v2", "dR0p32017v2", "newDM2017v2",
           # "deepTau2017v1",
           "deepTau2017v2p1",
           "deepTau2018v2p5",
           "deepTau2026v2p5",
           # "DPFTau_2016_v0",
           # "DPFTau_2016_v1",
           # "againstEle2018",
           ]
tauIdEmbedder = tauIdConfig.TauIDEmbedder(process, debug = False,
                    updatedTauName = updatedTauName,
                    toKeep = toKeep)
tauIdEmbedder.runTauID()
#Another tau collection with updated tauIDs
postfix = "Ver2"
tauIdEmbedder2 = tauIdConfig.TauIDEmbedder(process, debug = False,
                    originalTauName = "slimmedTaus", #one can run on top of other collection than default "slimmedTaus"
                    updatedTauName = updatedTauName+postfix,
                    postfix = postfix, # defaut "", specify non-trivial postfix if tool is run more than one time
                    toKeep = toKeep)
tauIdEmbedder2.runTauID()

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
process.out.outputCommands.append("keep *_"+updatedTauName+postfix+"_*_*")

# Path and EndPath definitions
process.p = cms.Path(
    process.rerunMvaIsolationSequence *
    getattr(process,updatedTauName)
    * getattr(process,"rerunMvaIsolationSequence"+postfix) *
    getattr(process,updatedTauName+postfix)
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
