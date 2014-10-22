import FWCore.ParameterSet.Config as cms

l1extraMuExtended = cms.EDProducer(
    "L1MuonParticleExtendedProducer",
    gmtROSource = cms.InputTag("simGmtDigis"),
    csctfSource = cms.InputTag("simCsctfTrackDigis"),
    writeAllCSCTFs = cms.bool(True)
    )
