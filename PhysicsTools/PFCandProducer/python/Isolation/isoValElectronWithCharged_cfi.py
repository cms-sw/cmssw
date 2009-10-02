import FWCore.ParameterSet.Config as cms

isoValElectronWithCharged = cms.EDProducer(
    "CandIsolatorFromDeposits",
    deposits = cms.VPSet(
    cms.PSet(
    src = cms.InputTag("isoDepElectronWithCharged"),
    deltaR = cms.double(0.5),
    weight = cms.string('1'),
    vetos = cms.vstring(),
    skipDefaultVeto = cms.bool(True),
    mode = cms.string('sum')
    )
    )
    )


