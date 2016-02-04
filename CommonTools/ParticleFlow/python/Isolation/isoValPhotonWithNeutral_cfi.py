import FWCore.ParameterSet.Config as cms

isoValPhotonWithNeutral = cms.EDProducer(
    "CandIsolatorFromDeposits",
    deposits = cms.VPSet(
    cms.PSet(
    src = cms.InputTag("isoDepPhotonWithNeutral"),
    deltaR = cms.double(0.4),
    weight = cms.string('1'), # 0.3333,
    vetos = cms.vstring('Threshold(0.5)'),
    skipDefaultVeto = cms.bool(True),
    mode = cms.string('sum')
    )
    )
    )


