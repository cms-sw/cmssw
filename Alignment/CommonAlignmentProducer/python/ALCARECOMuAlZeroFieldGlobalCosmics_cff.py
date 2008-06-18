import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlZeroFieldGlobalCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOMuAlZeroFieldGlobalCosmicsHLT.andOr = True
ALCARECOMuAlZeroFieldGlobalCosmicsHLT.HLTPaths = ["CandHLTTrackerCosmics", "CandHLTTrackerCosmicsCoTF", "HLT_IsoMu11", "HLT_Mu15_L1Mu7"]

ALCARECOMuAlZeroFieldGlobalCosmics = cms.EDFilter("ZeroFieldGlobalMuonBuilder",
                                                  inputTracker = cms.InputTag("cosmictrackfinderP5"),
                                                  inputMuon = cms.InputTag("cosmicMuons"),
                                                  minTrackerHits = cms.int32(0),
                                                  minMuonHits = cms.int32(0),
                                                  minPdot = cms.double(0.99),
                                                  minDdotP = cms.double(0.99),
                                                  debuggingHistograms = cms.untracked.bool(False),
                                                  )

seqALCARECOMuAlZeroFieldGlobalCosmics = cms.Sequence(ALCARECOMuAlZeroFieldGlobalCosmicsHLT + ALCARECOMuAlZeroFieldGlobalCosmics)
