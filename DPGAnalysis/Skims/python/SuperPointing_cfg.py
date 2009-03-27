import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
  '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v1/0002/0A12CE23-D7F9-DD11-819E-00E081348D21.root'),
                            secondaryFileNames = cms.untracked.vstring(
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/578/085EFED4-E5AB-DD11-9ACA-001617C3B6FE.root')
)

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.12 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/SuperPointing_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT SuperPointing skim')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRAFT_ALL_V4::All' 
process.prefer("GlobalTag")

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")


process.cosmicMuonsBarrelOnlyFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                   SALabel = cms.string("cosmicMuonsBarrelOnly"),
                                                   PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                   radius = cms.double(10.0),
                                                   maxZ = cms.double(50.0)
                                                   )

process.cosmicMuonsFilter = cms.EDFilter("HLTMuonPointingFilter",
                                         SALabel = cms.string("cosmicMuons"),
                                         PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                         radius = cms.double(10.0),
                                         maxZ = cms.double(50.0)
                                         )

process.cosmicMuons1LegFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                       SALabel = cms.string("cosmicMuons1Leg"),
                                                       PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                       radius = cms.double(10.0),
                                                       maxZ = cms.double(50.0)
                                                       )

process.globalCosmicMuonsBarrelOnlyFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                         SALabel = cms.string("globalCosmicMuonsBarrelOnly"),
                                                         PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                         radius = cms.double(10.0),
                                                         maxZ = cms.double(50.0)
                                                         )

process.cosmictrackfinderP5Filter = cms.EDFilter("HLTMuonPointingFilter",
                                                 SALabel = cms.string("cosmictrackfinderP5"),
                                                 PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                 radius = cms.double(10.0),
                                                 maxZ = cms.double(50.0)
                                                 )

process.globalCosmicMuonsFilter = cms.EDFilter("HLTMuonPointingFilter",
                                               SALabel = cms.string("globalCosmicMuons"),
                                               PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                               radius = cms.double(10.0),
                                               maxZ = cms.double(50.0)
                                               )

process.rsWithMaterialTracksP5Filter = cms.EDFilter("HLTMuonPointingFilter",
                                                    SALabel = cms.string("rsWithMaterialTracksP5"),
                                                    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                    radius = cms.double(10.0),
                                                    maxZ = cms.double(50.0)
                                                    )

process.globalCosmicMuons1LegFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                             SALabel = cms.string("globalCosmicMuons1Leg"),
                                                             PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                             radius = cms.double(10.0),
                                                             maxZ = cms.double(50.0)
                                                             )

process.ctfWithMaterialTracksP5Filter = cms.EDFilter("HLTMuonPointingFilter",
                                                     SALabel = cms.string("ctfWithMaterialTracksP5"),
                                                     PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                     radius = cms.double(10.0),
                                                     maxZ = cms.double(50.0)
                                                     )


process.cosmicMuonsBarrelOnlyPath = cms.Path(process.cosmicMuonsBarrelOnlyFilter)
process.cosmicMuonsPath = cms.Path(process.cosmicMuonsFilter)
process.cosmicMuons1LegPath = cms.Path(process.cosmicMuons1LegFilter)
process.globalCosmicMuonsBarrelOnlyPath = cms.Path(process.globalCosmicMuonsBarrelOnlyFilter)
process.cosmictrackfinderP5Path = cms.Path(process.cosmictrackfinderP5Filter)
process.globalCosmicMuonsPath = cms.Path(process.globalCosmicMuonsFilter)
process.rsWithMaterialTracksP5Path = cms.Path(process.rsWithMaterialTracksP5Filter)
process.globalCosmicMuons1LegPath = cms.Path(process.globalCosmicMuons1LegFilter)
process.ctfWithMaterialTracksP5Path = cms.Path(process.ctfWithMaterialTracksP5Filter)



process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cosmicMuonsBarrelOnlyPath',
                                                                                            'cosmicMuonsPath',
                                                                                            'cosmicMuons1LegPath',
                                                                                            'globalCosmicMuonsBarrelOnlyPath',
                                                                                            'cosmictrackfinderP5Path',
                                                                                            'globalCosmicMuonsPath',
                                                                                            'rsWithMaterialTracksP5Path',
                                                                                            'globalCosmicMuons1LegPath',
                                                                                            'ctfWithMaterialTracksP5Path')),                               
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('RAW-RECO'),
                                         filterName = cms.untracked.string('SuperPointing')),
                               fileName = cms.untracked.string('superPointing.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)



