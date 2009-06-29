import FWCore.ParameterSet.Config as cms

muonIsolatorFromDeposits = cms.EDProducer(
    "CandIsolatorFromDeposits",
    deposits = cms.VPSet(
    cms.PSet(
    src = cms.InputTag("isoMuonWithCharged"),
    deltaR = cms.double(0.5),
    weight = cms.string('1'),
    vetos = cms.vstring('0.01',
#                        'Threshold(2.0)'),
                        'Threshold(999999.0)'),
    skipDefaultVeto = cms.bool(True),
    mode = cms.string('sum')
    )
    )
    )


