import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL Zmuu for HO:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcazmumu_cfi import *
from Calibration.HcalAlCaRecoProducers.isoMuonHLT_cfi import *
seqALCARECOHcalCalZMuMu = cms.Sequence(isoMuonHLT*ALCARECOHcalCalZMuMu)

# foo bar baz
# uRVVOV8b1pbhG
