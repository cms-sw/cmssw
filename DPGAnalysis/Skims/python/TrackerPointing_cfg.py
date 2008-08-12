import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/data/CRUZET3/Cosmics/RECO/CRUZET3_V2P_v3/0060/226F5F00-3451-DD11-9688-000423D9853C.root')
                            )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRUZET3_V2P::All'
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

process.out = cms.OutputModule("PoolOutputModule",
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cosmicMuonsBarrelOnlyTkPath',
                                                                                            'cosmictrackfinderP5TkCntPath',
                                                                                            'ctfWithMaterialTracksP5TkCntPath',
                                                                                            'rsWithMaterialTracksP5TkCntPath')),
                               fileName = cms.untracked.string('trackerPointing.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)
