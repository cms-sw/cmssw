import FWCore.ParameterSet.Config as cms

from RecoHI.HiEgammaAlgos.HiIsolationCommonParameters_cff import *

isoC1 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(1),
    y  = cms.double(0),
)

isoC2 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(2),
    y  = cms.double(0),
)

isoC3 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(3),
    y  = cms.double(0),
)

isoC4 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(4),
    y  = cms.double(0),
)

isoC5 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(5),
    y  = cms.double(0),
)

isoCC1 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(1),
    y  = cms.double(0),
)

isoCC2 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(2),
    y  = cms.double(0),
)

isoCC3 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(3),
    y  = cms.double(0),
)

isoCC4 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(4),
    y  = cms.double(0),
)

isoCC5 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(5),
    y  = cms.double(0),
)

isoR1 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(1),
    y  = cms.double(0),
)

isoR2 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(2),
    y  = cms.double(0),
)

isoR3 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(3),
    y  = cms.double(0),
)

isoR4 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(4),
    y  = cms.double(0),
)

isoR5 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(5),
    y  = cms.double(0),
)

isoCR1 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(1),
    y  = cms.double(0),
)

isoCR2 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(2),
    y  = cms.double(0),
)

isoCR3 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(3),
    y  = cms.double(0),
)

isoCR4 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(4),
    y  = cms.double(0),
)

isoCR5 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(5),
    y  = cms.double(0),
)


hiEcalIsolation = cms.Sequence(isoC1+isoC2+isoC3+isoC4+isoC5)
hiEcalIsolationBckSubtracted = cms.Sequence(isoCC1+isoCC2+isoCC3+isoCC4+isoCC5)
hiHcalIsolation = cms.Sequence(isoR1+isoR2+isoR3+isoR4+isoR5)
hiHcalIsolationBckSubtracted = cms.Sequence(isoCR1+isoCR2+isoCR3+isoCR4+isoCR5)

hiCaloIsolation = cms.Sequence(hiEcalIsolation+hiHcalIsolation)
hiCaloIsolationBckSubtracted = cms.Sequence(hiEcalIsolationBckSubtracted+hiHcalIsolationBckSubtracted)

hiCaloIsolationAll = cms.Sequence(hiCaloIsolation+hiCaloIsolationBckSubtracted)


