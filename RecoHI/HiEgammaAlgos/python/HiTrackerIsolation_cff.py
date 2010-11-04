import FWCore.ParameterSet.Config as cms

from RecoHI.HiEgammaAlgos.HiIsolationCommonParameters_cff import *

isoT11 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(1),
    y  = cms.double(1),
    photons = cms.InputTag("cleanPhotons"),
)

isoT12 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(1),
    y  = cms.double(2),
    photons = cms.InputTag("cleanPhotons"),
)

isoT13 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(1),
    y  = cms.double(3),
    photons = cms.InputTag("cleanPhotons"),
)

isoT14 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(1),
    y  = cms.double(4),
    photons = cms.InputTag("cleanPhotons"),
)

isoT21 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(2),
    y  = cms.double(1),
    photons = cms.InputTag("cleanPhotons"),
)

isoT22 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(2),
    y  = cms.double(2),
    photons = cms.InputTag("cleanPhotons"),
)

isoT23 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(2),
    y  = cms.double(3),
    photons = cms.InputTag("cleanPhotons"),
)

isoT24 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(2),
    y  = cms.double(4),
    photons = cms.InputTag("cleanPhotons"),
)

isoT31 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(3),
    y  = cms.double(1),
    photons = cms.InputTag("cleanPhotons"),
)

isoT32 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(3),
    y  = cms.double(2),
    photons = cms.InputTag("cleanPhotons"),
)

isoT33 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(3),
    y  = cms.double(3),
    photons = cms.InputTag("cleanPhotons"),
)

isoT34 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(3),
    y  = cms.double(4),
    photons = cms.InputTag("cleanPhotons"),
)

isoT41 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(4),
    y  = cms.double(1),
    photons = cms.InputTag("cleanPhotons"),
)

isoT42 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(4),
    y  = cms.double(2),
    photons = cms.InputTag("cleanPhotons"),
)

isoT43 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(4),
    y  = cms.double(3),
    photons = cms.InputTag("cleanPhotons"),
)

isoT44 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(4),
    y  = cms.double(4),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR11 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(1),
    y  = cms.double(1),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR12 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(1),
    y  = cms.double(2),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR13 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(1),
    y  = cms.double(3),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR14 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(1),
    y  = cms.double(4),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR21 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(2),
    y  = cms.double(1),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR22 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(2),
    y  = cms.double(2),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR23 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(2),
    y  = cms.double(3),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR24 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(2),
    y  = cms.double(4),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR31 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(3),
    y  = cms.double(1),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR32 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(3),
    y  = cms.double(2),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR33 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(3),
    y  = cms.double(3),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR34 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(3),
    y  = cms.double(4),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR41 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(4),
    y  = cms.double(1),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR42 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(4),
    y  = cms.double(2),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR43 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(4),
    y  = cms.double(3),
    photons = cms.InputTag("cleanPhotons"),
)

isoDR44 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(4),
    y  = cms.double(4),
    photons = cms.InputTag("cleanPhotons"),
)

hiTrackCountingIsolation = cms.Sequence(isoT11+isoT12+isoT13+isoT14+isoT21+isoT22+isoT23+isoT24+isoT31+isoT32+isoT33+isoT34+isoT41+isoT42+isoT43+isoT44)
hiTrackVetoConeIsolation = cms.Sequence(isoDR11+isoDR12+isoDR13+isoDR14+isoDR21+isoDR22+isoDR23+isoDR24+isoDR31+isoDR32+isoDR33+isoDR34+isoDR41+isoDR42+isoDR43+isoDR44)

hiTrackerIsolation = cms.Sequence(hiTrackCountingIsolation+hiTrackVetoConeIsolation)


