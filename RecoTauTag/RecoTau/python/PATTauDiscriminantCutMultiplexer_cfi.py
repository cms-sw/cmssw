'''

Multiplex a cut on a PATTauDiscriminator using another PATTauDiscriminator as the
index.

Used by the anti-electron MVA, which needs to choose what cut to apply on the
MVA output depending on what the category is.

'''

import FWCore.ParameterSet.Config as cms

patTauDiscriminantCutMultiplexer = cms.EDProducer(
    "PATTauDiscriminantCutMultiplexer",
    PATTauProducer = cms.InputTag("fixme"),
    toMultiplex = cms.InputTag("fixme"),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string("and"),
        decayMode = cms.PSet(
            Producer = cms.InputTag("fixme"),
            cut = cms.double(0.)
            )
        ),
    loadMVAfromDB = cms.bool(True),
    inputFileName = cms.FileInPath("RecoTauTag/RecoTau/data/emptyMVAinputFile"), # the filename for MVA if it is not loaded from DB
    mvaOutput_normalization = cms.string(""), # the special value for not using a string parameter is empty string ""
    # it's the same value as the atribute of this plugin class is initialized with anyway
    # and throughout configs this parameter is everywhere set to non-empty value

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
    workingPoints = cms.vstring(),
    verbosity = cms.int32(0)
)
