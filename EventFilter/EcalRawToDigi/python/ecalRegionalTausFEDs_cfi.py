import FWCore.ParameterSet.Config as cms

ecalRegionalTausFEDs = cms.EDProducer("EcalRawToRecHitRoI",
    sourceTag = cms.InputTag("EcalRawToRecHitFacility"),
    JetJobPSet = cms.VPSet(cms.PSet(
        Source = cms.InputTag("l1extraParticles","Tau"),
        regionPhiMargin = cms.double(1.0),
        Ptmin = cms.double(20.0),
        regionEtaMargin = cms.double(1.0)
    )),
    type = cms.string('jet')
)


