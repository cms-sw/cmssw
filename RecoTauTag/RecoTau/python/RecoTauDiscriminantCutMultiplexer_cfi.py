'''

Multiplex a cut on a PFTauDiscriminator using another PFTauDiscriminator as the
index.

Used by the anti-electron MVA, which needs to choose what cut to apply on the
MVA output depending on what the category is.

'''

import FWCore.ParameterSet.Config as cms

recoTauDiscriminantCutMultiplexer = cms.EDProducer(
    "RecoTauDiscriminantCutMultiplexer",
    PFTauProducer = cms.InputTag("fixme"),
    toMultiplex = cms.InputTag("fixme"),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string("and"),
        decayMode = cms.PSet(
            Producer = cms.InputTag("fixme"),
            cut = cms.double(0.)
            )
        ),
    key = cms.InputTag("fixme"), # a discriminator
    loadMVAfromDB = cms.bool(True),
    inputFileName = cms.FileInPath("RecoTauTag/RecoTau/test/dummyMVAinputFile"), # the filename for MVA if it is not loaded from DB
    mvaOutput_normalization = cms.string(""), # empty basicaly does nothing: it rewrites the attribute which is initialized as empty string
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.double(0.5),
        ),
        cms.PSet(
            category = cms.uint32(1),
            cut = cms.double(0.2),
        ),
    ),
    verbosity = cms.int32(0)
)
