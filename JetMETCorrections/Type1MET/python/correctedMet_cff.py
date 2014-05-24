import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
caloMetT1 = cms.EDProducer(
    "CorrectedCaloMETProducer",
    src = cms.InputTag('caloMetM'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrCaloMetType1', 'type1')
        ),
)   

##____________________________________________________________________________||
caloMetT1T2 = cms.EDProducer(
    "CorrectedCaloMETProducer",
    src = cms.InputTag('caloMetM'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrCaloMetType1', 'type1'),
        cms.InputTag('corrCaloMetType2')
    ),
)   

##____________________________________________________________________________||
pfMetT0rt = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrack'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtT1 = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrack'),
        cms.InputTag('corrPfMetType1', 'type1'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtT1T2 = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrackForType2'),
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetType2'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtT2 = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrackForType2'),
        cms.InputTag('corrPfMetType2'),
    ),
)   

##____________________________________________________________________________||
pfMetT0pc = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0PfCand'),
    ),
)   

##____________________________________________________________________________||
pfMetT0pcT1 = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0PfCand'),
        cms.InputTag('corrPfMetType1', 'type1')
    ),
)   

##____________________________________________________________________________||
pfMetT1 = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType1', 'type1')
    ),
)   

##____________________________________________________________________________||
pfMetT1T2 = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetType2'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtTxy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrack'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtT1Txy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrack'),
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtT1T2Txy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrackForType2'),
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetType2'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtT2Txy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrackForType2'),
        cms.InputTag('corrPfMetType2'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT0pcTxy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0PfCand'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT0pcT1Txy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0PfCand'),
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT1Txy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT1T2Txy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetType2'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetMultCorr = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag("corrPfMetMult",  "e"),
        cms.InputTag("corrPfMetMult",  "egammaHFInnerMostRingsMinus"),
        cms.InputTag("corrPfMetMult",  "egammaHFInnerMostRingsPlus"),
        cms.InputTag("corrPfMetMult",  "egammaHFMinus"),
        cms.InputTag("corrPfMetMult",  "egammaHFPlus"),
        cms.InputTag("corrPfMetMult",  "gammaBarrel"),
        cms.InputTag("corrPfMetMult",  "gammaEndcapMinus"),
        cms.InputTag("corrPfMetMult",  "gammaEndcapPlus"),
        cms.InputTag("corrPfMetMult",  "gammaForwardMinus"),
        cms.InputTag("corrPfMetMult",  "gammaForwardPlus"),
        cms.InputTag("corrPfMetMult",  "h"),
        cms.InputTag("corrPfMetMult",  "h0Barrel"),
        cms.InputTag("corrPfMetMult",  "h0EndcapMinus"),
        cms.InputTag("corrPfMetMult",  "h0EndcapPlus"),
        cms.InputTag("corrPfMetMult",  "hHFInnerMostRingsMinus"),
        cms.InputTag("corrPfMetMult",  "hHFInnerMostRingsPlus"),
        cms.InputTag("corrPfMetMult",  "hHFMinus"),
        cms.InputTag("corrPfMetMult",  "hHFPlus")
    )
)   

