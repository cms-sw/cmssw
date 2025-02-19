import FWCore.ParameterSet.Config as cms

from RecoHI.HiEgammaAlgos.HiIsolationCommonParameters_cff import *

isoT11 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(1),
    y  = cms.double(1),
)

isoT12 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(1),
    y  = cms.double(2),
)

isoT13 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(1),
    y  = cms.double(3),
)

isoT14 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(1),
    y  = cms.double(4),
)

isoT21 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(2),
    y  = cms.double(1),
)

isoT22 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(2),
    y  = cms.double(2),
)

isoT23 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(2),
    y  = cms.double(3),
)

isoT24 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(2),
    y  = cms.double(4),
)

isoT31 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(3),
    y  = cms.double(1),
)

isoT32 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(3),
    y  = cms.double(2),
)

isoT33 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(3),
    y  = cms.double(3),
)

isoT34 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(3),
    y  = cms.double(4),
)

isoT41 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(4),
    y  = cms.double(1),
)

isoT42 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(4),
    y  = cms.double(2),
)

isoT43 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(4),
    y  = cms.double(3),
)

isoT44 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("Txy"),
    x  = cms.double(4),
    y  = cms.double(4),
)

isoDR11 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(1),
    y  = cms.double(1),
)

isoDR12 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(1),
    y  = cms.double(2),
)

isoDR13 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(1),
    y  = cms.double(3),
)

isoDR14 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(1),
    y  = cms.double(4),
)

isoDR21 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(2),
    y  = cms.double(1),
)

isoDR22 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(2),
    y  = cms.double(2),
)

isoDR23 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(2),
    y  = cms.double(3),
)

isoDR24 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(2),
    y  = cms.double(4),
)

isoDR31 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(3),
    y  = cms.double(1),
)

isoDR32 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(3),
    y  = cms.double(2),
)

isoDR33 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(3),
    y  = cms.double(3),
)

isoDR34 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(3),
    y  = cms.double(4),
)

isoDR41 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(4),
    y  = cms.double(1),
)

isoDR42 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(4),
    y  = cms.double(2),
)

isoDR43 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(4),
    y  = cms.double(3),
)

isoDR44 = cms.EDProducer("HiEgammaIsolationProducer",
    isolationInputParameters,
    mode  = cms.string("noBackgroundSubtracted"),
    iso  = cms.string("dRxy"),
    x  = cms.double(4),
    y  = cms.double(4),
)


hiTrackCountingIsolation = cms.Sequence(isoT11+isoT12+isoT13+isoT14+isoT21+isoT22+isoT23+isoT24+isoT31+isoT32+isoT33+isoT34+isoT41+isoT42+isoT43+isoT44)
hiTrackVetoConeIsolation = cms.Sequence(isoDR11+isoDR12+isoDR13+isoDR14+isoDR21+isoDR22+isoDR23+isoDR24+isoDR31+isoDR32+isoDR33+isoDR34+isoDR41+isoDR42+isoDR43+isoDR44)

hiTrackerIsolation = cms.Sequence(hiTrackCountingIsolation+hiTrackVetoConeIsolation)
