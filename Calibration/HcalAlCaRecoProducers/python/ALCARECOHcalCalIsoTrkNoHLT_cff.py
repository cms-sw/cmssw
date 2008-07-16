import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaisotrk_cfi import *
IsoProd.CheckHLTMatch = False

seqALCARECOHcalCalIsoTrk = cms.Sequence(IsoProd)


