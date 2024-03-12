import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaHcalHBHEMuonProducer_cfi import *
from Calibration.HcalAlCaRecoProducers.alcaHcalHBHEMuonFilter_cfi import *
from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import *

seqALCARECOHcalCalHBHEMuonProducerFilter = cms.Sequence(alcaHcalHBHEMuonProducer * alcaHcalHBHEMuonFilter)




# foo bar baz
# 7jRcEFHlFs3IV
# 75CrUvi6ADutR
