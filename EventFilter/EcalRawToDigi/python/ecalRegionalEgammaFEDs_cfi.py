import FWCore.ParameterSet.Config as cms

ecalRegionalEgammaFEDs = cms.EDFilter("EcalRawToRecHitRoI",
    sourceTag = cms.InputTag("EcalRawToRecHitFacility"),
    EmJobPSet = cms.VPSet(cms.PSet(
        Source = cms.InputTag("l1extraParticles","Isolated"),
        regionPhiMargin = cms.double(0.4),
        Ptmin = cms.double(5.0),
        regionEtaMargin = cms.double(0.25)
    ), cms.PSet(
        Source = cms.InputTag("l1extraParticles","NonIsolated"),
        regionPhiMargin = cms.double(0.4),
        Ptmin = cms.double(5.0),
        regionEtaMargin = cms.double(0.25)
    )),
    type = cms.string('egamma')
)


