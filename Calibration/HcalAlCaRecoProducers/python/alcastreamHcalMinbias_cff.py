import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcaminbias_cfi import *
seqAlcastreamHcalMinbias = cms.Sequence(MinProd)

