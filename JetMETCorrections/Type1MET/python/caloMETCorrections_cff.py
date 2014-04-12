import FWCore.ParameterSet.Config as cms

# load jet energy correction parameters
from JetMETCorrections.Configuration.JetCorrectionServices_cff import *

#--------------------------------------------------------------------------------
# produce Type 1 + 2 MET corrections for CaloJets
caloJetMETcorr = cms.EDProducer("CaloJetMETcorrInputProducer",
    src = cms.InputTag('ak5CaloJets'),
    jetCorrLabel = cms.string("ak5CaloL2L3"), # NOTE: use "ak5CaloL2L3" for MC / "ak5CaloL2L3Residual" for Data
    jetCorrEtaMax = cms.double(9.9),
    type1JetPtThreshold = cms.double(20.0),
    skipEM = cms.bool(True),
    skipEMfractionThreshold = cms.double(0.90),
    srcMET = cms.InputTag('corMetGlobalMuons')
)                                         
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# compute sum of muon corrections
muonCaloMETcorr = cms.EDProducer("MuonMETcorrInputProducer",
    src = cms.InputTag('muons'),
    srcMuonCorrections = cms.InputTag('muonMETValueMapProducer', 'muCorrData')
)                                 
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# use MET corrections to produce Type 1 / Type 1 + 2 corrected CaloMET objects
caloType1CorrectedMet = cms.EDProducer("CorrectedCaloMETProducer",
    src = cms.InputTag('corMetGlobalMuons'),
    applyType1Corrections = cms.bool(True),
    srcType1Corrections = cms.VInputTag(
        cms.InputTag('caloJetMETcorr', 'type1')
    ),
    applyType2Corrections = cms.bool(False)
)   

caloType1p2CorrectedMet = cms.EDProducer("CorrectedCaloMETProducer",
    src = cms.InputTag('corMetGlobalMuons'),
    applyType1Corrections = cms.bool(True),
    srcType1Corrections = cms.VInputTag(
        cms.InputTag('caloJetMETcorr', 'type1')
    ),
    applyType2Corrections = cms.bool(True),
    srcUnclEnergySums = cms.VInputTag(
        cms.InputTag('caloJetMETcorr', 'type2'),
        cms.InputTag('muonCaloMETcorr') # NOTE: use 'muonCaloMETcorr' for 'corMetGlobalMuons', do **not** use it for 'met' !!
    ),
    type2CorrFormula = cms.string("A + B*TMath::Exp(-C*x)"),
    type2CorrParameter = cms.PSet(
        A = cms.double(2.0),
        B = cms.double(1.3),
        C = cms.double(0.1)                                             
    )
)   
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# define sequence to run all modules
produceCaloMETCorrections = cms.Sequence(
    caloJetMETcorr
   * muonCaloMETcorr
   * caloType1CorrectedMet
   * caloType1p2CorrectedMet
)
#--------------------------------------------------------------------------------
