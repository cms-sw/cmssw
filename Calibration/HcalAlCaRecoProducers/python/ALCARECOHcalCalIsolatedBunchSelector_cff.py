import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
#AlCaReco filtering for HCAL isotrk:isolated bunch:
#-------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaIsolatedBunchSelector_cfi import *

seqALCARECOHcalCalIsolatedBunchSelector = cms.Sequence(AlcaIsolatedBunchSelector)
