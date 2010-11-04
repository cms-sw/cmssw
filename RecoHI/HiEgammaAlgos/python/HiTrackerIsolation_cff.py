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
