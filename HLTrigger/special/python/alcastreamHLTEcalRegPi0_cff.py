import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for pi0 calibration:
#------------------------------------------------
# create sequence for rechit filtering for pi0 calibration
from HLTrigger.special.alCaPi0HLTRegRecHits_cfi import *
seqAlcastreamEcalPi0 = cms.Sequence(alCaPi0RegRecHits)

