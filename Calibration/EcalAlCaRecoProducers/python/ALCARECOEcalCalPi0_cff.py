import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for pi0 calibration:
#------------------------------------------------
# create sequence for rechit filtering for pi0 calibration
from Calibration.EcalAlCaRecoProducers.alCaPi0RecHits_cfi import *
seqALCARECOEcalCalPi0 = cms.Sequence(alCaPi0RecHits)

