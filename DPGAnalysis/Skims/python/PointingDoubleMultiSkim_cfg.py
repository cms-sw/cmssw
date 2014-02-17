import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/CRUZET3/CMSSW_2_1_2/src/DPGAnalysis/Skims/python/reco_50908_210_CRZT210_V1P.root')
                            )

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/PointingDoubleMultiSkim_cfg.py,v $'),
    annotation = cms.untracked.string('CRUZET4 DoubleMuonPointing skim')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRZT210_V1::All' 
process.prefer("GlobalTag")

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")


################ Tracker Pointing ################ 

process.cosmicMuonsBarrelOnlyTkFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                     SALabel = cms.string("cosmicMuonsBarrelOnly"),
                                                     PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                     radius = cms.double(90.0),
                                                     maxZ = cms.double(130.0)
                                                     )

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

process.cosmicMuonsBarrelOnlyTkPath = cms.Path(process.cosmicMuonsBarrelOnlyTkFilter)
process.cosmictrackfinderP5TkCntPath = cms.Path(process.cosmictrackfinderP5TkCntFilter)
process.ctfWithMaterialTracksP5TkCntPath = cms.Path(process.ctfWithMaterialTracksP5TkCntFilter)
process.rsWithMaterialTracksP5TkCntPath = cms.Path(process.rsWithMaterialTracksP5TkCntFilter)

process.out1 = cms.OutputModule("PoolOutputModule",
                                SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cosmicMuonsBarrelOnlyTkPath',
                                                                                             'cosmictrackfinderP5TkCntPath',
                                                                                             'ctfWithMaterialTracksP5TkCntPath',
                                                                                             'rsWithMaterialTracksP5TkCntPath')),
                                dataset = cms.untracked.PSet(
			                  dataTier = cms.untracked.string('RECO'),
                                          filterName = cms.untracked.string('TrackerPointing')),
                                fileName = cms.untracked.string('trackerPointing.root')
                                )


################ Multi Muon ################ 


process.multiCosmicMuonFilter = cms.EDFilter("TrackCountFilter",
                                             src = cms.InputTag('cosmicMuonsBarrelOnly'),
                                             minNumber = cms.uint32(5) 
                                             )

process.multiLHCMuonFilter = cms.EDFilter("TrackCountFilter",
                                          src = cms.InputTag('lhcStandAloneMuonsBarrelOnly'),
                                          minNumber = cms.uint32(5) 
                                          )

process.multiCosmicMuonPath = cms.Path(process.multiCosmicMuonFilter)
process.multiLHCMuonPath = cms.Path(process.multiLHCMuonFilter)

process.out2 = cms.OutputModule("PoolOutputModule",
                                SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('multiCosmicMuonPath',
                                                                                             'multiLHCMuonPath')),
                                dataset = cms.untracked.PSet(
			                  dataTier = cms.untracked.string('RECO'),
                                          filterName = cms.untracked.string('multiCosmicMuon')),
                                fileName = cms.untracked.string('multiMuon.root')
                                )


################ Double Muon ################ 


process.doubleMuonFilter = cms.EDFilter("TrackCountFilter",
                                        src = cms.InputTag('cosmicMuonsBarrelOnly'),
                                        minNumber = cms.uint32(2) 
                                        )

process.doubleMuonPath = cms.Path(process.doubleMuonFilter)

process.out3 = cms.OutputModule("PoolOutputModule",
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('doubleMuonPath')),
                               dataset = cms.untracked.PSet(
			        	 dataTier = cms.untracked.string('RECO'),
                                	 filterName = cms.untracked.string('doubleMuonPath')),
                               fileName = cms.untracked.string('doubleMuon.root')
                               )

################ 



process.this_is_the_end = cms.EndPath(process.out1+process.out2+process.out3)
