import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:78F01A3E-1F90-DD11-A216-000423D99160.root')
                            #secondaryFileNames = cms.untracked.vstring('file:raw_50908_210_CRZT210_V1P.root')                            
                            )

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/SuperPointing_and_pixcluster_cfg.py,v $'),
    annotation = cms.untracked.string('CRUZET4 SuperPointing skim')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5000))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'auto:run3_data_prompt'
process.prefer("GlobalTag")

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

process.cosmicPixelClusterFilter = cms.EDFilter("PixelCountFilter",
                                                      src = cms.InputTag('siPixelClusters'),
                                                      minNumber = cms.uint32(2) 
                                                      )

process.cosmicMuonsBarrelOnlyFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                   SALabel = cms.InputTag("cosmicMuonsBarrelOnly"),
                                                   PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                   radius = cms.double(10.0),
                                                   maxZ = cms.double(50.0)
                                                   )

process.cosmicMuonsFilter = cms.EDFilter("HLTMuonPointingFilter",
                                         SALabel = cms.InputTag("cosmicMuons"),
                                         PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                         radius = cms.double(10.0),
                                         maxZ = cms.double(50.0)
                                         )

process.cosmicMuons1LegBarrelOnlyFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                       SALabel = cms.InputTag("cosmicMuons1LegBarrelOnly"),
                                                       PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                       radius = cms.double(10.0),
                                                       maxZ = cms.double(50.0)
                                                       )

process.globalCosmicMuonsBarrelOnlyFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                         SALabel = cms.InputTag("globalCosmicMuonsBarrelOnly"),
                                                         PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                         radius = cms.double(10.0),
                                                         maxZ = cms.double(50.0)
                                                         )

process.cosmictrackfinderP5Filter = cms.EDFilter("HLTMuonPointingFilter",
                                                 SALabel = cms.InputTag("cosmictrackfinderP5"),
                                                 PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                 radius = cms.double(10.0),
                                                 maxZ = cms.double(50.0)
                                                 )

process.globalCosmicMuonsFilter = cms.EDFilter("HLTMuonPointingFilter",
                                               SALabel = cms.InputTag("globalCosmicMuons"),
                                               PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                               radius = cms.double(10.0),
                                               maxZ = cms.double(50.0)
                                               )

process.rsWithMaterialTracksP5Filter = cms.EDFilter("HLTMuonPointingFilter",
                                                    SALabel = cms.InputTag("rsWithMaterialTracksP5"),
                                                    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                    radius = cms.double(10.0),
                                                    maxZ = cms.double(50.0)
                                                    )

process.globalCosmicMuons1LegBarrelOnlyFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                             SALabel = cms.InputTag("globalCosmicMuons1LegBarrelOnly"),
                                                             PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                             radius = cms.double(10.0),
                                                             maxZ = cms.double(50.0)
                                                             )

process.ctfWithMaterialTracksP5Filter = cms.EDFilter("HLTMuonPointingFilter",
                                                     SALabel = cms.InputTag("ctfWithMaterialTracksP5"),
                                                     PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                     radius = cms.double(10.0),
                                                     maxZ = cms.double(50.0)
                                                     )


process.cosmicPixelClusterPath = cms.Path(process.cosmicPixelClusterFilter)
process.cosmicMuonsBarrelOnlyPath = cms.Path(process.cosmicMuonsBarrelOnlyFilter)
process.cosmicMuonsPath = cms.Path(process.cosmicMuonsFilter)
process.cosmicMuons1LegBarrelOnlyPath = cms.Path(process.cosmicMuons1LegBarrelOnlyFilter)
process.globalCosmicMuonsBarrelOnlyPath = cms.Path(process.globalCosmicMuonsBarrelOnlyFilter)
process.cosmictrackfinderP5Path = cms.Path(process.cosmictrackfinderP5Filter)
process.globalCosmicMuonsPath = cms.Path(process.globalCosmicMuonsFilter)
process.rsWithMaterialTracksP5Path = cms.Path(process.rsWithMaterialTracksP5Filter)
process.globalCosmicMuons1LegBarrelOnlyPath = cms.Path(process.globalCosmicMuons1LegBarrelOnlyFilter)
process.ctfWithMaterialTracksP5Path = cms.Path(process.ctfWithMaterialTracksP5Filter)



process.out = cms.OutputModule("PoolOutputModule",
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cosmicPixelClusterPath',
                                                                                            'cosmicMuonsBarrelOnlyPath',
                                                                                            'cosmicMuonsPath',
                                                                                            'cosmicMuons1LegBarrelOnlyPath',
                                                                                            'globalCosmicMuonsBarrelOnlyPath',
                                                                                            'cosmictrackfinderP5Path',
                                                                                            'globalCosmicMuonsPath',
                                                                                            'rsWithMaterialTracksP5Path',
                                                                                            'globalCosmicMuons1LegBarrelOnlyPath',
                                                                                            'ctfWithMaterialTracksP5Path')),                               
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('RECO'),
                                         filterName = cms.untracked.string('SuperPointingPixelCluster')),
                               fileName = cms.untracked.string('superPointingPixelCluster.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)
