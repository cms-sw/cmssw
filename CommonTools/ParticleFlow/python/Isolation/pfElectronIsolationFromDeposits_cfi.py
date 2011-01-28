import FWCore.ParameterSet.Config as cms

pfElectronIsolationFromDeposits = cms.EDProducer(
    "CandIsolatorFromDeposits",
    deposits = cms.VPSet(
    cms.PSet(
    src = cms.InputTag("isoDepElectronWithCharged"),
    deltaR = cms.double(0.5),
    weight = cms.string('1'),
    vetos = cms.vstring('0.01',
                        'Threshold(0.0)'),
    skipDefaultVeto = cms.bool(True),
    mode = cms.string('sum')
    )
    )
    )


