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
phase2 = False

process = cms.Process('TauID')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
if phase2:
    process.load('Configuration.Geometry.GeometryExtended2026D97Reco_cff')
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T25', '')
    inputfile = '/store/mc/Phase2Spring21DRMiniAOD/TTbar_TuneCP5_14TeV-pythia8/MINIAODSIM/PU200Phase2D80_113X_mcRun4_realistic_T25_v1_ext1-v1/280000/04e6741c-489a-4fed-9e0c-d7703c274b5a.root'
else:
    process.load('Configuration.Geometry.GeometryRecoDB_cff')
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')
    inputfile = '/store/mc/RunIISummer20UL18MiniAOD/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v11_L1v1-v2/00000/009636D7-07B2-DB49-882D-C251FD62CCE7.root'

# Input source
process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring(
    # File from dataset TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8
    inputfile
))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(eventsToProcess) )

# Add new TauIDs
import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
toKeep = [ "2017v2", "dR0p32017v2", "newDM2017v2",
           # "deepTau2017v1",
           "deepTau2017v2p1",
           "deepTau2018v2p5",
           # "DPFTau_2016_v0",
           # "DPFTau_2016_v1",
           "againstEle2018",
           ]
if phase2:
    toKeep = [ "newDMPhase2v1",
               # "deepTau2018v2p5",
               "deepTau2026v2p5",
               "againstElePhase2v1",
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

# Adapt to old phase2 input samples where slimmedElectronsHGC are called slimmedElectronsFromMultiCl
if phase2:
    process.mergedSlimmedElectronsForTauId.src = ["slimmedElectrons","slimmedElectronsFromMultiCl"]

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
