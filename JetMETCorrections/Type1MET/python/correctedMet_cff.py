import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
caloMetT1 = cms.EDProducer(
    "AddCorrectionsToCaloMET",
    src = cms.InputTag('caloMetM'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrCaloMetType1', 'type1')
        ),
)   

##____________________________________________________________________________||
caloMetT1T2 = cms.EDProducer(
    "AddCorrectionsToCaloMET",
    src = cms.InputTag('caloMetM'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrCaloMetType1', 'type1'),
        cms.InputTag('corrCaloMetType2')
    ),
)   

##____________________________________________________________________________||
pfMetT0rt = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrack'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtT1 = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrack'),
        cms.InputTag('corrPfMetType1', 'type1'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtT1T2 = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrackForType2'),
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetType2'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtT2 = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrackForType2'),
        cms.InputTag('corrPfMetType2'),
    ),
)   

##____________________________________________________________________________||
pfMetT0pc = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0PfCand'),
    ),
)   

##____________________________________________________________________________||
pfMetT0pcT1 = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0PfCand'),
        cms.InputTag('corrPfMetType1', 'type1')
    ),
)   

##____________________________________________________________________________||
pfMetT1 = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType1', 'type1')
    ),
)   

##____________________________________________________________________________||
pfMetT1T2 = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetType2'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtTxy = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrack'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtT1Txy = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrack'),
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT0rtT1T2Txy = cms.EDProducer(
    "AddCorrectionsToPFMET",
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
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0RecoTrackForType2'),
        cms.InputTag('corrPfMetType2'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT0pcTxy = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0PfCand'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT0pcT1Txy = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0PfCand'),
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT1Txy = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
pfMetT1T2Txy = cms.EDProducer(
    "AddCorrectionsToPFMET",
    src = cms.InputTag('pfMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType1', 'type1'),
        cms.InputTag('corrPfMetType2'),
        cms.InputTag('corrPfMetShiftXY'),
    ),
)   

##____________________________________________________________________________||
