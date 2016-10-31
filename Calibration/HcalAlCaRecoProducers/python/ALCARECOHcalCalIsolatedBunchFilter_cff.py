import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
#AlCaReco filtering for HCAL isotrk:isolated bunch:
#-------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaIsolatedBunchFilter_cfi import *

seqALCARECOHcalCalIsolatedBunchFilter = cms.Sequence(AlcaIsolatedBunchFilter)
