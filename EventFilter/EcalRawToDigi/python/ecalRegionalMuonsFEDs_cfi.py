import FWCore.ParameterSet.Config as cms

ecalRegionalMuonsFEDs = cms.EDFilter("EcalRawToRecHitRoI",
    sourceTag = cms.InputTag("EcalRawToRecHitFacility"),
    MuJobPSet = cms.PSet(
        Source = cms.InputTag("l1extraParticles"),
        regionPhiMargin = cms.double(1.0),
        Ptmin = cms.double(0.0),
        regionEtaMargin = cms.double(1.0)
    ),
    type = cms.string('muon')
)


