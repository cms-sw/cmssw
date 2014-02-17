import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/D266D139-D871-DE11-A709-001D09F28F0C.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/CA27788D-E871-DE11-8B46-001D09F276CF.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/AC5633B2-D471-DE11-9B3A-001D09F252F3.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/9CD957E7-D071-DE11-B6AE-001D09F252F3.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/94BF68F7-D171-DE11-902B-000423D986A8.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/7838FE1E-C771-DE11-9FD5-000423D98950.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/56632803-DD71-DE11-BAF5-000423D9870C.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/42A67CB9-E971-DE11-AA86-001D09F252F3.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/407225D3-D071-DE11-809B-001D09F297EF.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/3E5E1CF0-D271-DE11-AC2B-000423D94700.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/2C57E916-D071-DE11-AF0E-001D09F24E39.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/228896A5-E571-DE11-A60B-001D09F2AF96.root')
)                            

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/PtMinSelectorNoSAMu_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT tk and global Muon Pt 50 skim')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))


process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR09_31X_V3P::All' 

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")



process.cosmictrackfinderP5PtFilter = cms.EDFilter("PtMinTrackSelector",
                                                   src = cms.InputTag("cosmictrackfinderP5"),
                                                   ptMin= cms.double(50.0),
                                                   filter = cms.bool(True)
                                                   )

process.globalCosmicMuonsPtFilter = cms.EDFilter("PtMinTrackSelector",
                                                 src = cms.InputTag("globalCosmicMuons"),
                                                 ptMin= cms.double(50.0),
                                                 filter = cms.bool(True)
                                                 )


process.globalCosmicMuons1LegPtFilter = cms.EDFilter("PtMinTrackSelector",
                                                     src = cms.InputTag("globalCosmicMuons1Leg"),
                                                     ptMin= cms.double(50.0),
                                                     filter = cms.bool(True)
                                                     )

process.ctfWithMaterialTracksP5PtFilter = cms.EDFilter("PtMinTrackSelector",
                                                       src = cms.InputTag("ctfWithMaterialTracksP5"),
                                                       ptMin= cms.double(50.0),
                                                       filter = cms.bool(True)
                                                       )


process.cosmictrackfinderP5Path = cms.Path(process.cosmictrackfinderP5PtFilter)
process.globalCosmicMuonsPath = cms.Path(process.globalCosmicMuonsPtFilter)
process.globalCosmicMuons1LegPath = cms.Path(process.globalCosmicMuons1LegPtFilter)
process.ctfWithMaterialTracksP5Path = cms.Path(process.ctfWithMaterialTracksP5PtFilter)



process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *',
                               'drop *_cosmictrackfinderP5PtFilter_*_*',
                               'drop *_globalCosmicMuonsPtFilter_*_*',
                               'drop *_globalCosmicMuons1LegPtFilter_*_*',
                               'drop *_ctfWithMaterialTracksP5PtFilter_*_*',
                               'drop *_MEtoEDMConverter_*_*'
				),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cosmictrackfinderP5Path',
                                                                                            'globalCosmicMuonsPath',
                                                                                            'globalCosmicMuons1LegPath',
                                                                                            'ctfWithMaterialTracksP5Path')),                               
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('RECO'),
                                         filterName = cms.untracked.string('GlobalTkMuonPt50')),
                               fileName = cms.untracked.string('/tmp/malgeri/PtMinSelectorNoSAMu.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)
