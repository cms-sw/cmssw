import FWCore.ParameterSet.Config as cms

l1tBmtfAlgoSelector = cms.EDProducer(
    'L1TBMTFAlgoSelector',
    # verbose = cms.untracked.bool(False),
    bmtfKalman = cms.InputTag("simKBmtfDigis:BMTF"),
    bmtfLegacy = cms.InputTag("simBmtfDigis:BMTF"),
    feds = cms.InputTag("rawDataCollector")
    )

