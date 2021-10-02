import FWCore.ParameterSet.Config as cms

# producer for alcadijets (HCAL gamma-jet)
import Calibration.HcalAlCaRecoProducers.alcaGammaJetProducer_cfi
GammaJetProd = Calibration.HcalAlCaRecoProducers.alcaGammaJetProducer_cfi.alcaGammaJetProducer.clone()

import Calibration.HcalAlCaRecoProducers.alcaGammaJetSelector_cfi
GammaJetFilter = Calibration.HcalAlCaRecoProducers.alcaGammaJetSelector_cfi.alcaGammaJetSelector.clone()



