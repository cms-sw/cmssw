import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaHcalIsotrkProducer_cff import *
from Calibration.HcalAlCaRecoProducers.alcaHcalIsotrkFilter_cfi import *
from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import *

seqALCARECOHcalCalIsoTrkProducerFilter = cms.Sequence(alcaHcalIsotrkProducer * alcaHcalIsotrkFilter)




