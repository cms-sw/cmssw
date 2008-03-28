import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for pi0 calibration:
#------------------------------------------------
# create sequence for rechit filtering for pi0 calibration
from Calibration.EcalAlCaRecoProducers.alCaPi0BCRecHits_cfi import *
seqAlcastreamEcalPi0BC = cms.Sequence(alCaPi0BCRecHits)

