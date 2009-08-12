import FWCore.ParameterSet.Config as cms

from RecoHI.HiEgammaAlgos.HiIsolationCommonParameters_cff import *

isoC1 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(1),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoC2 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(2),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoC3 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(3),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoC4 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(4),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoC5 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(5),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoCC1 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(1),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoCC2 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(2),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoCC3 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(3),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoCC4 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(4),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoCC5 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Cx"),
    x  = cms.double(5),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoR1 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(1),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoR2 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(2),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoR3 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(3),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoR4 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(4),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoR5 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(5),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoCR1 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(1),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoCR2 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(2),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoCR3 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(3),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoCR4 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(4),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)

isoCR5 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("BackgroundSubtracted"),
    iso  = cms.string("Rx"),
    x  = cms.double(5),
    y  = cms.double(0),
    photons = cms.InputTag("photons"),
)


hiEcalIsolation = cms.Sequence(isoC1+isoC2+isoC3+isoC4+isoC5)
hiEcalIsolationBckSubtracted = cms.Sequence(isoCC1+isoCC2+isoCC3+isoCC4+isoCC5)
hiHcalIsolation = cms.Sequence(isoR1+isoR2+isoR3+isoR4+isoR5)
hiHcalIsolationBckSubtracted = cms.Sequence(isoCR1+isoCR2+isoCR3+isoCR4+isoCR5)

hiCaloIsolation = cms.Sequence(hiEcalIsolation+hiHcalIsolation)
hiCaloIsolationBckSubtracted = cms.Sequence(hiEcalIsolationBckSubtracted+hiHcalIsolationBckSubtracted)

hiCaloIsolationAll = cms.Sequence(hiCaloIsolation+hiCaloIsolationBckSubtracted)


