import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

#process.load("DPGAnalysis/Skims/ecalSkim_cfi")
process.skimming = cms.EDFilter("EcalSkim",
    #cosmic cluster energy threshold in GeV
    energyCutEB = cms.untracked.double(2.0),
    energyCutEE = cms.untracked.double(2.0),
    endcapClusterCollection = cms.InputTag("cosmicSuperClusters","CosmicEndcapSuperClusters"),
    barrelClusterCollection = cms.InputTag("cosmicSuperClusters","CosmicBarrelSuperClusters")
)

process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(0),
    debugFlag = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(
  '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v1/0002/0A12CE23-D7F9-DD11-819E-00E081348D21.root')
)

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.4 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/ecalSkim_fromRECO_cfg.py,v $'),
    annotation = cms.untracked.string('CRUZET4 ecalSkim_fromRECO skim')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ecalSkim.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RECO'),
    	      filterName = cms.untracked.string('ecalSkim_fromRECO')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)

process.p = cms.Path(process.skimming)
process.e = cms.EndPath(process.out)

