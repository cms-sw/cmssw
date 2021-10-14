import FWCore.ParameterSet.Config as cms

# producer for alcaminbisas (HCAL minimum bias)

import Calibration.HcalAlCaRecoProducers.alcaEcalHcalReadoutsProducer_cfi
HcalNoiseProd = Calibration.HcalAlCaRecoProducers.alcaEcalHcalReadoutsProducer_cfi.alcaEcalHcalReadoutsProducer.clone()


