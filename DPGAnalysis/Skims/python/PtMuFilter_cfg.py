import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
  '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v1/0002/0A12CE23-D7F9-DD11-819E-00E081348D21.root'
    )
)                            

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/PtMuFilter_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT Pt_50 Skim')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRAFT_ALL_V8::All' 
process.prefer("GlobalTag")

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")



process.cosmictrackfinderP5PtFilter = cms.EDFilter("MuonPtFilter",
                                                   SALabel = cms.string("cosmictrackfinderP5"),
                                                   minPt= cms.double(50.0)
                                                   )

process.globalCosmicMuonsPtFilter = cms.EDFilter("MuonPtFilter",
                                                 SALabel = cms.string("globalCosmicMuons"),
                                                 minPt= cms.double(50.0)
                                                 )


process.globalCosmicMuons1LegPtFilter = cms.EDFilter("MuonPtFilter",
                                                     SALabel = cms.string("globalCosmicMuons1Leg"),
                                                     minPt= cms.double(50.0)
                                                     )

process.ctfWithMaterialTracksP5PtFilter = cms.EDFilter("MuonPtFilter",
                                                       SALabel = cms.string("ctfWithMaterialTracksP5"),
                                                       minPt= cms.double(50.0)
                                                       )

process.cosmicMuons1LegPtFilter = cms.EDFilter("MuonPtFilter",
                                               SALabel = cms.string("cosmicMuons1Leg"),
                                               minPt= cms.double(50.0)
                                               )

process.cosmictrackfinderP5Path = cms.Path(process.cosmictrackfinderP5PtFilter)
process.globalCosmicMuonsPath = cms.Path(process.globalCosmicMuonsPtFilter)
process.globalCosmicMuons1LegPath = cms.Path(process.globalCosmicMuons1LegPtFilter)
process.ctfWithMaterialTracksP5Path = cms.Path(process.ctfWithMaterialTracksP5PtFilter)
process.cosmicMuons1LegPath = cms.Path(process.cosmicMuons1LegPtFilter)



process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cosmictrackfinderP5Path',
                                                                                            'globalCosmicMuonsPath',
                                                                                            'globalCosmicMuons1LegPath',
                                                                                            'ctfWithMaterialTracksP5Path',
                                                                                            'cosmicMuons1LegPath')),                               
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('RECO'),
                                         filterName = cms.untracked.string('MuonPt50')),
                               fileName = cms.untracked.string('PtMuFilter.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)
