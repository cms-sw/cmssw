import FWCore.ParameterSet.Config as cms

ecalRegionalCandFEDs = cms.EDProducer("EcalListOfFEDSProducer",
    sourceTag = cms.InputTag("EcalRawToRecHitFacility"),
    CandJobPSet = cms.VPSet(cms.PSet(
        bePrecise = cms.bool(False),
        propagatorNameToBePrecise = cms.string(''),
        epsilon = cms.double(0.01),
        regionPhiMargin = cms.double(1.0),
        cType = cms.string('l1muon'),
        Source = cms.InputTag("l1extraParticles"),
        Ptmin = cms.double(0.0),
        regionEtaMargin = cms.double(1.0)
    ), cms.PSet(
        bePrecise = cms.bool(False),
        propagatorNameToBePrecise = cms.string(''),
        epsilon = cms.double(0.01),
        regionPhiMargin = cms.double(1.0),
        cType = cms.string('l1jet'),
        Source = cms.InputTag("l1extraParticles","Central"),
        Ptmin = cms.double(50.0),
        regionEtaMargin = cms.double(1.0)
    ), cms.PSet(
        bePrecise = cms.bool(False),
        propagatorNameToBePrecise = cms.string(''),
        epsilon = cms.double(0.01),
        regionPhiMargin = cms.double(1.0),
        cType = cms.string('view'),
        Source = cms.InputTag("l1extraParticles","Tau"),
        Ptmin = cms.double(50.0),
        regionEtaMargin = cms.double(1.0)
    ), cms.PSet(
        bePrecise = cms.bool(False),
        propagatorNameToBePrecise = cms.string(''),
        epsilon = cms.double(0.01),
        regionPhiMargin = cms.double(0.5),
        cType = cms.string('chargedcandidate'),
        Source = cms.InputTag("hltL2MuonCandidates"),
        Ptmin = cms.double(0.0),
        regionEtaMargin = cms.double(0.5)
    ), cms.PSet(
        bePrecise = cms.bool(False),
        propagatorNameToBePrecise = cms.string(''),
        epsilon = cms.double(0.01),
        regionPhiMargin = cms.double(0.5),
        cType = cms.string('chargedcandidate'),
        Source = cms.InputTag("hltL3MuonCandidates"),
        Ptmin = cms.double(0.0),
        regionEtaMargin = cms.double(0.5)
    )),
    type = cms.string('candidate')
)


