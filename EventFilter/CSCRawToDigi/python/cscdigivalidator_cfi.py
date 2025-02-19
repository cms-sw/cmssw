import FWCore.ParameterSet.Config as cms

cscDigiValidator = cms.EDAnalyzer('CSCDigiValidator',
                                  inputStrip = cms.untracked.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
                                  inputWire = cms.untracked.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
                                  inputComp = cms.untracked.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
                                  inputCLCT = cms.untracked.InputTag("simCscTriggerPrimitiveDigis",""),
                                  inputALCT = cms.untracked.InputTag("simCscTriggerPrimitiveDigis",""),
                                  inputCorrLCT = cms.untracked.InputTag("simCscTriggerPrimitiveDigis",""),
                                  inputCSCTF = cms.untracked.InputTag("simCsctfTrackDigis",""),
                                  inputCSCTFStubs = cms.untracked.InputTag("simCsctfTrackDigis",""),
                                  repackStrip = cms.untracked.InputTag("muonCSCDigis","MuonCSCStripDigi"),
                                  repackWire = cms.untracked.InputTag("muonCSCDigis","MuonCSCWireDigi"),
                                  repackComp = cms.untracked.InputTag("muonCSCDigis","MuonCSCComparatorDigi"),
                                  repackCLCT = cms.untracked.InputTag("muonCSCDigis","MuonCSCCLCTDigi"),
                                  repackALCT = cms.untracked.InputTag("muonCSCDigis","MuonCSCALCTDigi"),
                                  repackCorrLCT = cms.untracked.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
                                  repackCSCTF = cms.untracked.InputTag("csctfDigis",""),
                                  repackCSCTFStubs = cms.untracked.InputTag("csctfDigis","DT"),
                                  applyStripReordering = cms.untracked.bool(True) #this does not work yet, me1a might be flipped if used on real data
)
