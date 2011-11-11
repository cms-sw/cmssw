import FWCore.ParameterSet.Config as cms

process = cms.Process("DISPLAY")

#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

### Expects test.root in current directory.
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring('file:test.root')
)

process.POurFilter = cms.EDFilter("RandomFilter",
                                    acceptRate = cms.untracked.double(1.0))
# process.maxEvents = cms.untracked.PSet(
#         input = cms.untracked.int32(1)
#         )

### For running on pre 3.6 files the current needed to determine the
### magnetic field is taken from Conditions DB.
# process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
### specify tag:
# process.GlobalTag.globaltag = 'START36_V10::All'
### or use auto-cond:
# from Configuration.AlCa.autoCond import autoCond
# process.GlobalTag.globaltag = autoCond['mc']
