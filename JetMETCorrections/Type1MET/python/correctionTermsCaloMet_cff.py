import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Configuration.JetCorrectors_cff import *

corrCaloMetType1 = cms.EDProducer(
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
muCaloMetCorr = cms.EDProducer("MuonMETcorrInputProducer",
    src = cms.InputTag('muons'),
    srcMuonCorrections = cms.InputTag('muonMETValueMapProducer', 'muCorrData')
)

##____________________________________________________________________________||
corrCaloMetType2 = cms.EDProducer(
    "Type2CorrectionProducer",
    srcUnclEnergySums = cms.VInputTag(
        cms.InputTag('corrCaloMetType1', 'type2'),
        cms.InputTag('muCaloMetCorr') # NOTE: use this for 'caloMetM', do **not** use it for 'met' !!
        ),
    type2CorrFormula = cms.string("A + B*TMath::Exp(-C*x)"),
    type2CorrParameter = cms.PSet(
        A = cms.double(2.0),
        B = cms.double(1.3),
        C = cms.double(0.1)
        )
    )

##____________________________________________________________________________||
correctionTermsCaloMetTask = cms.Task(
    ak4CaloL2L3CorrectorTask, # NOTE: use "ak4CaloL2L3CorrectorTask" for MC / "ak4CaloL2L3ResidualCorrectorTask" for Data
    ak4CaloL2L3ResidualCorrectorTask,
    corrCaloMetType1,
    muCaloMetCorr,
    corrCaloMetType2
    )

correctionTermsCaloMet = cms.Sequence(correctionTermsCaloMetTask)
