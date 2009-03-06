import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/CRUZET3/CMSSW_2_1_2/src/DPGAnalysis/Skims/python/reco_50908_210_CRZT210_V1P.root')
                            )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRZT210_V1::All' 
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
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('multiCosmicMuonPath',
                                                                                            'multiLHCMuonPath')),
                               fileName = cms.untracked.string('multiMuon.root')
                               )


process.multiCosmicMuonPath = cms.Path(process.multiCosmicMuonFilter)
process.multiLHCMuonPath = cms.Path(process.multiLHCMuonFilter)

process.this_is_the_end = cms.EndPath(process.out)
