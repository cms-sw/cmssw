import FWCore.ParameterSet.Config as cms
import RecoLocalCalo.HcalRecProducers.hfsimplereco_cfi as _mod

hfQIE10Reco = _mod.hfsimplereco.clone(
    digiLabel = "simHcalUnsuppressedDigis:HFQIE10DigiCollection",
    Subdetector = 'HFQIE10',
    firstSample = 2,
    samplesToAdd = 1
)


