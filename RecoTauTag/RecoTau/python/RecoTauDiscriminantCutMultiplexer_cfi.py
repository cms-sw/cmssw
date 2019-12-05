'''

Multiplex a cut on a PFTauDiscriminator using another PFTauDiscriminator as the
index.

Used by the anti-electron MVA, which needs to choose what cut to apply on the
MVA output depending on what the category is.

'''

import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.recoTauDiscriminantCutMultiplexerDefault_cfi import recoTauDiscriminantCutMultiplexerDefault

recoTauDiscriminantCutMultiplexer = recoTauDiscriminantCutMultiplexerDefault.clone(
    PFTauProducer = cms.InputTag("fixme"),
    toMultiplex = cms.InputTag("fixme"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("fixme"),
        ),
        cms.PSet(
            category = cms.uint32(1),
            cut = cms.string("fixme"),
        ),
    )
)
