import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcaisotrk_cfi import *
from Calibration.HcalAlCaRecoProducers.isoHLT_cfi import *
seqALCARECOHcalCalIsoTrk = cms.Sequence(isoHLT*IsoProd)

