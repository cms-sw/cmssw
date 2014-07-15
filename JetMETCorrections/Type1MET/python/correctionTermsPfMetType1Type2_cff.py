import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Type1MET.pfMETCorrections_cff import *

##____________________________________________________________________________||
corrPfMetType1 = pfJetMETcorr.clone()

##____________________________________________________________________________||
corrPfMetType2 = cms.EDProducer(
    "Type2CorrectionProducer",
    srcUnclEnergySums = cms.VInputTag(
        cms.InputTag('corrPfMetType1', 'type2'),
        cms.InputTag('corrPfMetType1', 'offset'),
        cms.InputTag('pfCandMETcorr')
    ),
    type2CorrFormula = cms.string("A"),
    type2CorrParameter = cms.PSet(
        A = cms.double(1.4)
        )
    )

##____________________________________________________________________________||
correctionTermsPfMetType1Type2 = cms.Sequence(
    pfJetsPtrForMetCorr +
    particleFlowPtrs +
    pfCandsNotInJetsPtrForMetCorr +
    pfCandsNotInJetsForMetCorr +
    pfCandMETcorr +
    corrPfMetType1 +
    corrPfMetType2
    )

##____________________________________________________________________________||
