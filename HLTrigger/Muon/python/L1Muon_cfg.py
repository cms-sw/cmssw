# The following comments couldn't be translated into the new config version:

#------------ Utilities ---------------------------------
# Minimal Message logger

# Starting point: drop everything

# Keep all objects created by this process

import FWCore.ParameterSet.Config as cms

process = cms.Process("PRODL1")
#------------ L1 Muon Filter -------------------------------------
process.load("HLTrigger.Muon.CommonModules_2x1033_cff")

import copy
process.load("HLTrigger.HLTfilters.hltLevel1GTSeed_cfi")

process.MuLevel1Seed = copy.deepcopy(process.hltLevel1GTSeed)
# Recover original digis if necessary
process.load("Configuration.EventContent.EventContent_cff")

process.source = cms.Source("PoolSource",
    #untracked vstring fileNames ={ 'file:hlt_data/Hto4mu_50.root' }
    #untracked vstring fileNames ={ 'rfio:/castor/cern.ch/user/j/jalcaraz/crab/Hto4mu_50.root' }
    maxEvents = cms.untracked.int32(10),
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/mc/2007/2/4/mc-onsel-120_PU_pp_muX-DIGI-RECO-NoPU/0002/007DD08A-CAB6-DB11-AEA0-001143D49026.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.L1MuL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("L1MuLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

process.OUTPUT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_*_*_PRODL1'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pl1sel')
    ),
    fileName = cms.untracked.string('L1Muons.root')
)

process.pl1sel = cms.Path(process.l1muonreco+process.L1MuLevel1Seed+process.L1MuL1Filtered)
process.outpath = cms.EndPath(process.OUTPUT)
process.MuLevel1Seed.L1SeedsLogicalExpression = 'L1_SingleMu3 OR L1_SingleMu5 OR L1_SingleMu7 OR L1_SingleMu10 OR L1_SingleMu14 OR L1_SingleMu20 OR 1_SingleMu25 OR L1_DoubleMu3 OR L1_TripleMu3'
process.OUTPUT.outputCommands.extend(process.SimG4CoreFEVT.outputCommands)
process.OUTPUT.outputCommands.extend(process.SimTrackerFEVT.outputCommands)
process.OUTPUT.outputCommands.extend(process.SimMuonFEVT.outputCommands)
process.OUTPUT.outputCommands.extend(process.SimCalorimetryFEVT.outputCommands)

