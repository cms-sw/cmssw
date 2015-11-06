import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Type1MET.caloMETCorrections_cff import *

##____________________________________________________________________________||
corrCaloMetType1 = caloJetMETcorr.clone()

##____________________________________________________________________________||
corrCaloMetType2 = cms.EDProducer(
    "Type2CorrectionProducer",
    srcUnclEnergySums = cms.VInputTag(
        cms.InputTag('corrCaloMetType1', 'type2'),
        cms.InputTag('muonCaloMETcorr') # NOTE: use 'muonCaloMETcorr' for 'corMetGlobalMuons', do **not** use it for 'met' !!
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
    muonCaloMETcorr +
    corrCaloMetType2
    )

##____________________________________________________________________________||
