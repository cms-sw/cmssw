'''

Multiplex a cut on a PFTauDiscriminator using another PFTauDiscriminator as the
index.

Used by the anti-electron MVA, which needs to choose what cut to apply on the
MVA output depending on what the category is.

'''

import FWCore.ParameterSet.Config as cms

recoTauDiscriminantSimpleCut = cms.EDProducer(
    "RecoTauDiscriminantSimpleCut",
    PFTauProducer = cms.InputTag("hpsPFTauProducer"),
 
    Prediscriminants = cms.PSet(
     BooleanOperator = cms.string('and'),
     decayMode = cms.PSet(
        Producer = cms.InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs"),
        cut = cms.double(0.5)
     )
    ),


    rawIsoPtSum  = cms.InputTag(''),
    cut = cms.double(0.),
    maximumSumPtCut = cms.double(6.0),
    applySumPtCut = cms.bool(False)
)
