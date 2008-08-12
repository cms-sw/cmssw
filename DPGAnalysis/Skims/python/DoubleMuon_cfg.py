import FWCore.ParameterSet.Config as cms

process = cms.Process("RecoMuon")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/data/CRUZET3/Cosmics/RECO/CRUZET3_V2P_v3/0060/226F5F00-3451-DD11-9688-000423D9853C.root')
                            )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))


process.load("RecoMuon.Configuration.MessageLogger_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRUZET3_V2P::All'
process.prefer("GlobalTag")

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")


process.doubleMuonFilter = cms.EDFilter("TrackCountFilter",
                                        src = cms.InputTag('cosmicMuonsBarrelOnly'),
                                        minNumber = cms.uint32(2) 
                                        )

process.doubleMuonPath = cms.Path(process.doubleMuonFilter)

process.out = cms.OutputModule("PoolOutputModule",
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('doubleMuonPath')),
                               fileName = cms.untracked.string('doubleMuon.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)


