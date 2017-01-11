import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
#labels used for the phi correction



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
pfMetTxy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetXYMult')
    ),
)   

##____________________________________________________________________________||
pfMetT0rtTxy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrack'),
        cms.InputTag('corrPfMetXYMult')
    ),
)   

##____________________________________________________________________________||
pfMetT0rtT1Txy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrack'),
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetXYMult')
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
        cms.InputTag('corrPfMetXYMult')
    ),
)   

##____________________________________________________________________________||
pfMetT0pcTxy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0PfCand'),
        cms.InputTag('corrPfMetXYMult')
    ),
)   

##____________________________________________________________________________||
pfMetT0pcT1Txy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0PfCand'),
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetXYMult')
    ),
)  

##____________________________________________________________________________||
pfMetT0pcT1T2Txy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0PfCand'),
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetType2'),
        cms.InputTag('corrPfMetXYMult')
    ),
)   

##____________________________________________________________________________||
pfMetT1Txy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetXYMult')
    ),
)  

##____________________________________________________________________________||
pfMetT1T2Txy = cms.EDProducer(
    "CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetType2'),
        cms.InputTag('corrPfMetXYMult')
    ),
)   
