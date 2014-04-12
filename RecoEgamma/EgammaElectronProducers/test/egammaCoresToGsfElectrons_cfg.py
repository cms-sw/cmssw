import FWCore.ParameterSet.Config as cms
import os
import dbs_discovery

from TrackingTools.Configuration.TrackingTools_cff import *

process = cms.Process("electrons")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.EventContent.EventContent_cff")

process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(1),
    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring()
)

process.source.fileNames.extend(dbs_discovery.search())
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep recoSuperClusters*_*_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_*_*_electrons', 
        'keep *HepMCProduct_*_*_*'),
    fileName = cms.untracked.string(os.environ['TEST_RECO_FILE'])
)

process.p = cms.Path(process.gsfElectrons)

process.outpath = cms.EndPath(process.out)
process.GlobalTag.globaltag = os.environ['TEST_GLOBAL_TAG']+'::All'



