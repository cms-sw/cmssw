import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaHcalHBHEMuonProducer_cfi import *
from Calibration.HcalAlCaRecoProducers.alcaHcalHEMuonFilter_cfi import *

seqALCARECOHcalCalHEMuonProducerFilter = cms.Sequence(alcaHcalHBHEMuonProducer * alcaHcalHEMuonFilter)




