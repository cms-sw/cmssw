import FWCore.ParameterSet.Config as cms

hltHcalMETNoiseFilter = cms.EDFilter("HLTHcalMETNoiseFilter",
    HcalNoiseRBXCollection = cms.InputTag("hcalnoise"),
    HcalNoiseSummary       = cms.InputTag("hcalnoise"),
    severity = cms.int32(1),

    # note that more than one filter can be applied at once
    useLooseFilter = cms.bool(True),
    useTightFilter = cms.bool(False),
    useHighLevelFilter = cms.bool(False),
    useCustomFilter = cms.bool(False),

    # parameters for custom filter
    # only used if useCustomFilter==True
    minE2Over10TS = cms.double(0.7),
    min25GeVHitTime = cms.double(-7.0),
    max25GeVHitTime = cms.double(6.0),
    maxZeros = cms.int32(8),
    maxHPDHits = cms.int32(16),
    maxRBXHits = cms.int32(72),
    minHPDEMF = cms.double(-9999999.),
    minRBXEMF = cms.double(-9999999.)
)
