import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
#labels used for the phi correction


##____________________________________________________________________________||
patMetTxy = cms.EDProducer(
    "CorrectedPatMETProducer",
    src = cms.InputTag('slimmedMETs'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetXYMultDB')
    ),
)   

