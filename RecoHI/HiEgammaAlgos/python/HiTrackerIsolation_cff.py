import FWCore.ParameterSet.Config as cms

from RecoHI.HiEgammaAlgos.HiIsolationCommonParameters_cff import *

isoT11 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(1),
    y  = cms.double(1),
    photons = cms.InputTag("photons"),
)

isoT12 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(1),
    y  = cms.double(2),
    photons = cms.InputTag("photons"),
)

isoT13 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(1),
    y  = cms.double(3),
    photons = cms.InputTag("photons"),
)

isoT14 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(1),
    y  = cms.double(4),
    photons = cms.InputTag("photons"),
)

isoT21 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(2),
    y  = cms.double(1),
    photons = cms.InputTag("photons"),
)

isoT22 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(2),
    y  = cms.double(2),
    photons = cms.InputTag("photons"),
)

isoT23 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(2),
    y  = cms.double(3),
    photons = cms.InputTag("photons"),
)

isoT24 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(2),
    y  = cms.double(4),
    photons = cms.InputTag("photons"),
)

isoT31 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(3),
    y  = cms.double(1),
    photons = cms.InputTag("photons"),
)

isoT32 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(3),
    y  = cms.double(2),
    photons = cms.InputTag("photons"),
)

isoT33 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(3),
    y  = cms.double(3),
    photons = cms.InputTag("photons"),
)

isoT34 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(3),
    y  = cms.double(4),
    photons = cms.InputTag("photons"),
)

isoT41 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(4),
    y  = cms.double(1),
    photons = cms.InputTag("photons"),
)

isoT42 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(4),
    y  = cms.double(2),
    photons = cms.InputTag("photons"),
)

isoT43 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(4),
    y  = cms.double(3),
    photons = cms.InputTag("photons"),
)

isoT44 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(4),
    y  = cms.double(4),
    photons = cms.InputTag("photons"),
)

isoDR11 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(1),
    y  = cms.double(1),
    photons = cms.InputTag("photons"),
)

isoDR12 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(1),
    y  = cms.double(2),
    photons = cms.InputTag("photons"),
)

isoDR13 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(1),
    y  = cms.double(3),
    photons = cms.InputTag("photons"),
)

isoDR14 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(1),
    y  = cms.double(4),
    photons = cms.InputTag("photons"),
)

isoDR21 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(2),
    y  = cms.double(1),
    photons = cms.InputTag("photons"),
)

isoDR22 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(2),
    y  = cms.double(2),
    photons = cms.InputTag("photons"),
)

isoDR23 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(2),
    y  = cms.double(3),
    photons = cms.InputTag("photons"),
)

isoDR24 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(2),
    y  = cms.double(4),
    photons = cms.InputTag("photons"),
)

isoDR31 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(3),
    y  = cms.double(1),
    photons = cms.InputTag("photons"),
)

isoDR32 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(3),
    y  = cms.double(2),
    photons = cms.InputTag("photons"),
)

isoDR33 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(3),
    y  = cms.double(3),
    photons = cms.InputTag("photons"),
)

isoDR34 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(3),
    y  = cms.double(4),
    photons = cms.InputTag("photons"),
)

isoDR41 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(4),
    y  = cms.double(1),
    photons = cms.InputTag("photons"),
)

isoDR42 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(4),
    y  = cms.double(2),
    photons = cms.InputTag("photons"),
)

isoDR43 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(4),
    y  = cms.double(3),
    photons = cms.InputTag("photons"),
)

isoDR44 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(4),
    y  = cms.double(4),
    photons = cms.InputTag("photons"),
)


hiTrackCountingIsolation = cms.Sequence(isoT11+isoT12+isoT13+isoT14+isoT21+isoT22+isoT23+isoT24+isoT31+isoT32+isoT33+isoT34+isoT41+isoT42+isoT43+isoT44)
hiTrackVetoConeIsolation = cms.Sequence(isoDR11+isoDR12+isoDR13+isoDR14+isoDR21+isoDR22+isoDR23+isoDR24+isoDR31+isoDR32+isoDR33+isoDR34+isoDR41+isoDR42+isoDR43+isoDR44)

hiTrackerIsolation = cms.Sequence(hiTrackCountingIsolation+hiTrackVetoConeIsolation)
