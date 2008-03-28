import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcadijets_cfi import *
from Calibration.HcalAlCaRecoProducers.dijetsHLT_cfi import *
seqALCARECOHcalCalDijets = cms.Sequence(dijetsHLT*DiJProd)

