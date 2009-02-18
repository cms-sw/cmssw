import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
#    'file:/afs/cern.ch/user/g/goys/scratch0/CMSSW_2_2_4/src/DPGAnalysis/Skims/python/superPointing_newdata.root'
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V8_ReReco-fnal-bigrun-TEST-v3/0000/0015445A-46F4-DD11-9697-00304867901A.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V8_ReReco-fnal-bigrun-TEST-v3/0000/027AF0D3-40F4-DD11-8D9A-003048679294.root'
    )
)                            

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.0 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/PtMinSelector_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT Pt skim')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(4000))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRAFT_ALL_V8::All' 
process.prefer("GlobalTag")

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
process.cosmicMuons1LegPtFilter = cms.EDFilter("PtMinTrackSelector",
                                               src = cms.InputTag("cosmicMuons1Leg"),
                                               ptMin= cms.double(50.0),
                                               filter = cms.bool(True)
                                               )


process.cosmictrackfinderP5Path = cms.Path(process.cosmictrackfinderP5PtFilter)
process.globalCosmicMuonsPath = cms.Path(process.globalCosmicMuonsPtFilter)
process.globalCosmicMuons1LegPath = cms.Path(process.globalCosmicMuons1LegPtFilter)
process.ctfWithMaterialTracksP5Path = cms.Path(process.ctfWithMaterialTracksP5PtFilter)
process.cosmicMuons1LegPath = cms.Path(process.cosmicMuons1LegPtFilter)



process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cosmictrackfinderP5Path',
                                                                                            'globalCosmicMuonsPath',
                                                                                            'globalCosmicMuons1LegPath',
                                                                                            'ctfWithMaterialTracksP5Path',
                                                                                            'cosmicMuons1LegPath')),                               
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('RECO'),
                                         filterName = cms.untracked.string('Pt')),
                               fileName = cms.untracked.string('PtMinSelector.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)
