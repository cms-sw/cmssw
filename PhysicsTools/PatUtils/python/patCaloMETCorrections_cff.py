import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectorsAllAlgos_cff import *

# load modules for producing Type 1 / Type 1 + 2 corrections for reco::PFMET objects

#--------------------------------------------------------------------------------
# produce "raw" (uncorrected) pat::MET of calo-type
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs
patCaloMet = patMETs.clone(
    metSource = cms.InputTag('caloMetM'),
    addMuonCorrections = cms.bool(False),
    genMETSource = cms.InputTag('genMetTrue')
)
#--------------------------------------------------------------------------------

patCaloMetType1Corr = cms.EDProducer(
    "CaloJetMETcorrInputProducer",
    src = cms.InputTag('ak4CaloJets'),
    jetCorrLabel = cms.InputTag("ak4CaloL2L3Corrector"), # NOTE: use "ak4CaloL2L3Corrector" for MC / "ak4CaloL2L3ResidualCorrector" for Data
    jetCorrEtaMax = cms.double(9.9),
    type1JetPtThreshold = cms.double(20.0),
    skipEM = cms.bool(True),
    skipEMfractionThreshold = cms.double(0.90),
    srcMET = cms.InputTag('caloMetM')
)

##____________________________________________________________________________||
patCaloMetMuCorr = cms.EDProducer("MuonMETcorrInputProducer",
    src = cms.InputTag('muons'),
    srcMuonCorrections = cms.InputTag('muonMETValueMapProducer', 'muCorrData')
)

##____________________________________________________________________________||
patCaloMetType2Corr = cms.EDProducer(
    "Type2CorrectionProducer",
    srcUnclEnergySums = cms.VInputTag(
        cms.InputTag('patCaloMetType1Corr', 'type2'),
        cms.InputTag('patCaloMetMuCorr') # NOTE: use this for 'corMetGlobalMuons', do **not** use it for 'met' !!
        ),
    type2CorrFormula = cms.string("A + B*exp(-C*x)"),
    type2CorrParameter = cms.PSet(
        A = cms.double(2.0),
        B = cms.double(1.3),
        C = cms.double(0.1)
        )
    )



#--------------------------------------------------------------------------------
# use MET corrections to produce Type 1 / Type 1 + 2 corrected PFMET objects
patCaloMetT1 = cms.EDProducer("CorrectedPATMETProducer",
    src = cms.InputTag('patCaloMet'),
    applyType1Corrections = cms.bool(True),
    srcType1Corrections = cms.VInputTag(
        cms.InputTag('patCaloMetType1Corr', 'type1'),
    ),
    applyType2Corrections = cms.bool(False)
)

patCaloMetT1T2 = cms.EDProducer("CorrectedPATMETProducer",
    src = cms.InputTag('patCaloMet'),
    applyType1Corrections = cms.bool(True),
    srcType1Corrections = cms.VInputTag(
        cms.InputTag('patCaloMetType1Corr', 'type1'),
    ),
    applyType2Corrections = cms.bool(True),
    srcUnclEnergySums = cms.VInputTag(
        cms.InputTag('patCaloMetType1Corr', 'type2' ),
    ),
    type2CorrFormula = cms.string("A + B*exp(-C*x)"),
    type2CorrParameter = cms.PSet(
        A = cms.double(2.0),
        B = cms.double(1.3),
        C = cms.double(0.1)
        )
)


##____________________________________________________________________________||
producePatCaloMETCorrectionsUnc = cms.Sequence(
    patCaloMet +
    ak4CaloL2L3CorrectorChain + # NOTE: use "ak4CaloL2L3CorrectorChain" for MC / "ak4CaloL2L3ResidualCorrectorChain" for Data
    patCaloMetType1Corr +
    patCaloMetMuCorr +
    patCaloMetType2Corr
    )

##____________________________________________________________________________||
