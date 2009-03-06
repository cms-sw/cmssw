import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
  '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v1/0002/0A12CE23-D7F9-DD11-819E-00E081348D21.root'),
                            )

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/MultiMuon_cfg.py,v $'),
    annotation = cms.untracked.string('CRUZET4 MultiMuon skim')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRAFT_ALL_V9::All' 
process.prefer("GlobalTag")

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

process.multiCosmicMuonFilter = cms.EDFilter("TrackCountFilter",
                                             src = cms.InputTag('cosmicMuonsBarrelOnly'),
                                             minNumber = cms.uint32(5) 
                                             )

process.multiLHCMuonFilter = cms.EDFilter("TrackCountFilter",
                                          src = cms.InputTag('lhcStandAloneMuonsBarrelOnly'),
                                          minNumber = cms.uint32(5) 
                                          )

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('multiCosmicMuonPath',
                                                                                            'multiLHCMuonPath')),
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('RECO'),
                                         filterName = cms.untracked.string('multiCosmicMuon')),
                               fileName = cms.untracked.string('multiMuon.root')
                               )


process.multiCosmicMuonPath = cms.Path(process.multiCosmicMuonFilter)
process.multiLHCMuonPath = cms.Path(process.multiLHCMuonFilter)

process.this_is_the_end = cms.EndPath(process.out)
