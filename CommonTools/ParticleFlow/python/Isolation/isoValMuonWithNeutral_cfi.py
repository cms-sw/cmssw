import FWCore.ParameterSet.Config as cms

isoValMuonWithNeutral = cms.EDProducer(
    "CandIsolatorFromDeposits",
    deposits = cms.VPSet(
    cms.PSet(
    src = cms.InputTag("isoDepMuonWithNeutral"),
    deltaR = cms.double(0.4),
    weight = cms.string('1'), # 0.33333
    vetos = cms.vstring('Threshold(0.5)'),
    skipDefaultVeto = cms.bool(True),
    mode = cms.string('sum')
    )
    )
    )


