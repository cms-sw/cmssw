import FWCore.ParameterSet.Config as cms

hiPFCandCleaner = cms.EDProducer(
    "HiPFCandCleaner",
    ptMin = cms.double(4.),
    absEtaMax = cms.double(2.),
    candidatesSrc = cms.InputTag("particleFlow")
    )
