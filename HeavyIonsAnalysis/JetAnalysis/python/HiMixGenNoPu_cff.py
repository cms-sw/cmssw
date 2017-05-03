import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.mixObjects_cfi import theMixObjects
from SimGeneral.MixingModule.mixPoolSource_cfi import *

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(0),
    minBunch = cms.int32(0), ## in terms of 25 nsec
    bunchspace = cms.int32(1), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),

    mixObjects = cms.PSet(
        mixHepMC = cms.PSet(
            input = cms.VInputTag(cms.InputTag("generator")),
            makeCrossingFrame = cms.untracked.bool(True),
            type = cms.string('HepMCProduct')
            )
        ),
)

mixGen = cms.Sequence(mix)


