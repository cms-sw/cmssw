import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcaisotrk_cfi import *
seqAlcastreamHcalIsotrk = cms.Sequence(alcaisotrk)

