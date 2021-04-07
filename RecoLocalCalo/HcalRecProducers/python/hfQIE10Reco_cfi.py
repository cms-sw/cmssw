import FWCore.ParameterSet.Config as cms
import RecoLocalCalo.HcalRecProducers.hosimplereco_cfi as _mod

hfQIE10Reco = _mod.hosimplereco.clone(
    correctionPhaseNS = 0.0,
    digiLabel = "simHcalUnsuppressedDigis:HFQIE10DigiCollection",
    Subdetector = 'HFQIE10',
    correctForPhaseContainment = False,
    correctForTimeslew = False,
    firstSample = 2,
    samplesToAdd = 1
)


