import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
#process.load("AuxCode.CheckTkCollection.CraftAlCaReco_cff")
#process.load("MyChecks.CheckTkCollection.Alca3Tesla_cff")
process.load("AuxCode.CheckTkCollection.Run123151_RECO_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.LhcTrackAnalyzer = cms.EDAnalyzer("LhcTrackAnalyzer",
                                          TrackCollectionTag = cms.InputTag("generalTracks"),
                                          PVtxCollectionTag = cms.InputTag("offlinePrimaryVertices"),
                                          OutputFileName = cms.string("LhcTrackAnalyzer.root"),
                                          Debug = cms.bool(False)
                                          )

process.p = cms.Path(process.LhcTrackAnalyzer)
