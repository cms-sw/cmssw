import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcaminbias_cfi import *
from Calibration.HcalAlCaRecoProducers.hcalminbiasHLT_cfi import *
seqALCARECOHcalCalMinBias = cms.Sequence(hcalminbiasHLT*MinProd)

