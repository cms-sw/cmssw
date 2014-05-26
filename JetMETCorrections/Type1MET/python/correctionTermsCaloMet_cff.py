import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Configuration.JetCorrectionServices_cff import *

corrCaloMetType1 = cms.EDProducer(
    "CaloJetMETcorrInputProducer",
    src = cms.InputTag('ak4CaloJets'),
    jetCorrLabel = cms.string("ak4CaloL2L3"), # NOTE: use "ak4CaloL2L3" for MC / "ak4CaloL2L3Residual" for Data
    jetCorrEtaMax = cms.double(9.9),
    type1JetPtThreshold = cms.double(20.0),
    skipEM = cms.bool(True),
    skipEMfractionThreshold = cms.double(0.90),
    srcMET = cms.InputTag('corMetGlobalMuons')
)

##____________________________________________________________________________||
muCaloMetCorr = cms.EDProducer("MuonMETcorrInputProducer",
    src = cms.InputTag('muons'),
    srcMuonCorrections = cms.InputTag('muonMETValueMapProducer', 'muCorrData')
)

##____________________________________________________________________________||
corrCaloMetType2 = cms.EDProducer(
    "Type2CorrectionProducer",
    srcUnclEnergySums = cms.VInputTag(
        cms.InputTag('corrCaloMetType1', 'type2'),
        cms.InputTag('muCaloMetCorr') # NOTE: use this for 'corMetGlobalMuons', do **not** use it for 'met' !!
        ),
    type2CorrFormula = cms.string("A + B*TMath::Exp(-C*x)"),
    type2CorrParameter = cms.PSet(
        A = cms.double(2.0),
        B = cms.double(1.3),
        C = cms.double(0.1)
        )
    )

##____________________________________________________________________________||
correctionTermsCaloMet = cms.Sequence(
    corrCaloMetType1 +
    muCaloMetCorr +
    corrCaloMetType2
    )

##____________________________________________________________________________||
