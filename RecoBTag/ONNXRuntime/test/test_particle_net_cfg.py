import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
#options.inputFiles = '/store/mc/RunIIFall17MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/90000/DCFE3F5F-AE42-E811-B6DB-008CFAF72A64.root'
options.inputFiles = 'file:/storage/local/data1/home/jduarte1/forPatrick/FFA0194D-1BBC-EF4F-9B8F-8FBED2C62FC8.root'
#options.inputFiles = '/store/mc/RunIISummer19UL17MiniAOD/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_mc2017_realistic_v6-v4/30000/FFA0194D-1BBC-EF4F-9B8F-8FBED2C62FC8.root'
#options.inputFiles = 'file:FFA0194D-1BBC-EF4F-9B8F-8FBED2C62FC8.root'
options.maxEvents = 100
options.parseArguments()

from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
process = cms.Process('PATtest',enableSonicTriton)

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1


## Options and Output Report
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## Source
process.source = cms.Source("PoolSource",
    fileNames=cms.untracked.vstring(options.inputFiles)
)
## Maximal Number of Events
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))

process.load("HeterogeneousCore.SonicTriton.TritonService_cff")
process.TritonService.verbose = True
# fallback server
process.TritonService.fallback.enable = True
process.TritonService.fallback.verbose = True
process.TritonService.servers.append(
    cms.PSet(
        name = cms.untracked.string("default"),
        #address = cms.untracked.string("prp-gpu-1.t2.ucsd.edu"),
        address = cms.untracked.string("ailab01.fnal.gov"),
        port = cms.untracked.uint32(8001),
    )
)


## Geometry and Detector Conditions (needed for a few patTuple production steps)
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic')
process.load("Configuration.StandardSequences.MagneticField_cff")

## Output Module Configuration (expects a path 'p')
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('patTuple.root'),
                               ## save only events passing the full path
                               #SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               ## save PAT output; you need a '*' to unpack the list of commands
                               ## 'patEventContent'
                               outputCommands = cms.untracked.vstring('drop *', *patEventContentNoCleaning )
                               )

patAlgosToolsTask = getPatAlgosToolsTask(process)
process.outpath = cms.EndPath(process.out, patAlgosToolsTask)

## and add them to the event content
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetJetTagsProbs as pfParticleNetJetTagsProbs
from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetSonicJetTagsProbs as pfParticleNetSonicJetTagsProbs

updateJetCollection(
   process,
   jetSource = cms.InputTag('slimmedJetsAK8'),
   pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
   svSource = cms.InputTag('slimmedSecondaryVertices'),
   rParam = 0.8,
   jetCorrections = ('AK8PFPuppi', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = pfParticleNetJetTagsProbs+pfParticleNetSonicJetTagsProbs
   )

from Configuration.EventContent.EventContent_cff import MINIAODSIMEventContent
process.out.outputCommands.append('keep *_slimmedJetsAK8*_*_*')
process.out.outputCommands.append('keep *_offlineSlimmedPrimaryVertices*_*_*')
process.out.outputCommands.append('keep *_slimmedSecondaryVertices*_*_*')
process.out.outputCommands.append('keep *_selectedPatJets*_*_*')
process.out.outputCommands.append('keep *_selectedUpdatedPatJets*_*_*')
process.out.outputCommands.append('keep *_updatedPatJets*_*_*')

process.out.fileName = 'test_particle_net_MINIAODSIM_noragged.root'


