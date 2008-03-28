import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcadijets_cfi import *
seqAlcastreamHcalDijets = cms.Sequence(DiJProd)

