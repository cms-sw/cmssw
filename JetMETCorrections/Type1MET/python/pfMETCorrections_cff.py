import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------------
from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import *

#--------------------------------------------------------------------------------
pfJetMETcorr = corrPfMetType1.clone()

#--------------------------------------------------------------------------------
from JetMETCorrections.Type1MET.correctionTermsPfMetType0RecoTrack_cff import *

#--------------------------------------------------------------------------------
# use MET corrections to produce Type 1 / Type 1 + 2 corrected PFMET objects
pfType1CorrectedMet = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    applyType0Corrections = cms.bool(False),
    srcCHSSums = cms.VInputTag(
        cms.InputTag('pfchsMETcorr', 'type0')
    ),
    type0Rsoft = cms.double(0.6),
    applyType1Corrections = cms.bool(True),
    srcType1Corrections = cms.VInputTag(
        cms.InputTag('pfJetMETcorr', 'type1')
    ),
    applyType2Corrections = cms.bool(False)
)

pfType1p2CorrectedMet = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    applyType0Corrections = cms.bool(False),
    srcCHSSums = cms.VInputTag(
        cms.InputTag('pfchsMETcorr', 'type0')
    ),
    type0Rsoft = cms.double(0.6),
    applyType1Corrections = cms.bool(True),
    srcType1Corrections = cms.VInputTag(
        cms.InputTag('pfJetMETcorr', 'type1')
    ),
    applyType2Corrections = cms.bool(True),
    srcUnclEnergySums = cms.VInputTag(
        cms.InputTag('pfJetMETcorr', 'type2'),
        cms.InputTag('pfJetMETcorr', 'offset'),
        cms.InputTag('pfCandMETcorr')
    ),
    type2CorrFormula = cms.string("A"),
    type2CorrParameter = cms.PSet(
        A = cms.double(1.4)
    )
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# define sequence to run all modules
producePFMETCorrections = cms.Sequence(
    pfJetsPtrForMetCorr
   * particleFlowPtrs
   * pfCandsNotInJetsPtrForMetCorr
   * pfCandsNotInJetsForMetCorr
   * pfJetMETcorr
   * pfCandMETcorr
   * pfchsMETcorr
   * pfType1CorrectedMet
   * pfType1p2CorrectedMet
)
#--------------------------------------------------------------------------------
