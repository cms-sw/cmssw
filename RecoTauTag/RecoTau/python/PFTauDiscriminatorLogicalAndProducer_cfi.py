'''

Select PFTaus that pass either all (option "and") or at least one (option "or") discriminators specified in "Prediscriminants" list.


'''

import FWCore.ParameterSet.Config as cms

PFTauDiscriminatorLogicalAndProducer = cms.EDProducer("PFTauDiscriminatorLogicalAndProducer",
    PFTauProducer = cms.InputTag("pfRecoTauProducer"),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string("and"), # pass all discriminats in the list
#         BooleanOperator = cms.string("or"), # pass at least one discriminat in the list
        discr1 = cms.PSet(
            Producer = cms.InputTag("pfRecoTauDiscriminationByIsolation"),
            cut = cms.double(0.5)
        ),
        discr2 = cms.PSet(
            Producer = cms.InputTag("pfRecoTauDiscriminationAgainstElectron"),
            cut = cms.double(0.5)
        )
    ),
    PassValue = cms.double(1.),
    FailValue = cms.double(0.)
)
