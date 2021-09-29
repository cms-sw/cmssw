import FWCore.ParameterSet.Config as cms

# producer for alcahbhemuon (HCAL with muons)

import Calibration.HcalAlCaRecoProducers.alcaHBHEMuonProducer_cfi
HBHEMuonProd = Calibration.HcalAlCaRecoProducers.alcaHBHEMuonProducer_cfi.alcaHBHEMuonProducer.clone()
