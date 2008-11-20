import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
       '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/122/D424EBA5-55A0-DD11-A8BF-000423D9853C.root',
       '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/122/C67EDF0D-49A0-DD11-9403-001617DBD332.root'),
                            secondaryFileNames = cms.untracked.vstring(
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/067/122/6E2601EC-3FA0-DD11-BA50-000423D986A8.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/067/122/C240B0B2-47A0-DD11-A6AD-001617C3B654.root') 
)                            
                            

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.9 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/TrackerPointing_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT TrackerPointing skim')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRAFT_V4P::All' 
process.prefer("GlobalTag")

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")




process.cosmictrackfinderP5TkCntFilter = cms.EDFilter("TrackCountFilter",
                                                      src = cms.InputTag('cosmictrackfinderP5'),
                                                      minNumber = cms.uint32(1) 
                                                      )

process.ctfWithMaterialTracksP5TkCntFilter = cms.EDFilter("TrackCountFilter",
                                                          src = cms.InputTag('ctfWithMaterialTracksP5'),
                                                          minNumber = cms.uint32(1) 
                                                          )

process.rsWithMaterialTracksP5TkCntFilter = cms.EDFilter("TrackCountFilter",
                                                         src = cms.InputTag('rsWithMaterialTracksP5'),
                                                         minNumber = cms.uint32(1) 
                                                         )

process.cosmicMuonsBarrelOnlyTkFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                     SALabel = cms.string("cosmicMuonsBarrelOnly"),
                                                     PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                     radius = cms.double(90.0),
                                                     maxZ = cms.double(130.0)
                                                     )

process.cosmicMuonsBarrelOnlyTkPath = cms.Path(process.cosmicMuonsBarrelOnlyTkFilter)
process.cosmictrackfinderP5TkCntPath = cms.Path(process.cosmictrackfinderP5TkCntFilter)
process.ctfWithMaterialTracksP5TkCntPath = cms.Path(process.ctfWithMaterialTracksP5TkCntFilter)
process.rsWithMaterialTracksP5TkCntPath = cms.Path(process.rsWithMaterialTracksP5TkCntFilter)


process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cosmicMuonsBarrelOnlyTkPath',
                                                                                            'cosmictrackfinderP5TkCntPath',
                                                                                            'ctfWithMaterialTracksP5TkCntPath',
                                                                                            'rsWithMaterialTracksP5TkCntPath')),
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('RAW-RECO'),
                                         filterName = cms.untracked.string('TrackingPointing')),
                               fileName = cms.untracked.string('trackerPointing.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)
