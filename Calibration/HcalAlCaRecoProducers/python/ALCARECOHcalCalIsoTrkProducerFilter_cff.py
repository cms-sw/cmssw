import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaHcalIsotrkProducer_cfi import *
from Calibration.HcalAlCaRecoProducers.alcaHcalIsotrkFilter_cfi import *

seqALCARECOHcalCalIsoTrkProducerFilter = cms.Sequence(alcaHcalIsotrkProducer * alcaHcalIsotrkFilter)




