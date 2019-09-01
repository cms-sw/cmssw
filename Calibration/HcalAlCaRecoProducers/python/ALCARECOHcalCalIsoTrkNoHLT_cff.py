import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaisotrk_cfi import *
IsoProd.CheckHLTMatch = False

seqALCARECOHcalCalIsoTrkNoHLT = cms.Sequence(alcaisotrk)


