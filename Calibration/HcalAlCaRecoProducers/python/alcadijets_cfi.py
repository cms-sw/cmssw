import FWCore.ParameterSet.Config as cms

# producer for alcadijet (HCAL dijet)
import Calibration.HcalAlCaRecoProducers.alcaDiJetsProducer_cfi
DiJetsProd = Calibration.HcalAlCaRecoProducers.alcaDiJetsProducer_cfi.alcaDiJetsProducer.clone()
