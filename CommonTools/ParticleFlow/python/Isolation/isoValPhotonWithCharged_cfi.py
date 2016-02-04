import FWCore.ParameterSet.Config as cms

isoValPhotonWithCharged = cms.EDProducer(
    "CandIsolatorFromDeposits",
    deposits = cms.VPSet(
    cms.PSet(
    src = cms.InputTag("isoDepPhotonWithCharged"),
    deltaR = cms.double(0.4),
    weight = cms.string('1'),
    vetos = cms.vstring(),
    skipDefaultVeto = cms.bool(True),
    mode = cms.string('sum')
    )
    )
    )


