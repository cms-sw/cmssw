import FWCore.ParameterSet.Config as cms

TauMcInfoProducer = cms.EDFilter("HLTTauMcInfo",
    UsePFTauMatching = cms.bool(True),
    PFTauProducer = cms.InputTag("pfRecoTauProducer"),
    BosonPID = cms.int32(23),
    PFTauDiscriminator = cms.InputTag("pfRecoTauDiscriminationByIsolation"),
    PtMin = cms.double(10.0),
    GenParticles = cms.InputTag("source"),
    # int32    BosonPID     = 37 //(H+)
    # int32    BosonPID     = 35 //(H0)
    # int32    BosonPID     = 36 //(A0)
    EtaMax = cms.double(2.5)
)


