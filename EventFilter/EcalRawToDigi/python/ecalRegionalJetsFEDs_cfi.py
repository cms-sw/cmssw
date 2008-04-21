import FWCore.ParameterSet.Config as cms

ecalRegionalJetsFEDs = cms.EDFilter("EcalRawToRecHitRoI",
    sourceTag = cms.InputTag("EcalRawToRecHitFacility"),
    JetJobPSet = cms.VPSet(cms.PSet(
        Source = cms.InputTag("l1extraParticles","Central"),
        regionPhiMargin = cms.double(1.0),
        Ptmin = cms.double(50.0),
        regionEtaMargin = cms.double(1.0)
    ), 
        cms.PSet(
            Source = cms.InputTag("l1extraParticles","Forward"),
            regionPhiMargin = cms.double(1.0),
            Ptmin = cms.double(50.0),
            regionEtaMargin = cms.double(1.0)
        ), 
        cms.PSet(
            Source = cms.InputTag("l1extraParticles","Tau"),
            regionPhiMargin = cms.double(1.0),
            Ptmin = cms.double(50.0),
            regionEtaMargin = cms.double(1.0)
        )),
    type = cms.string('jet')
)


