import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL HO:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcahomuon_cfi import *
from Calibration.HcalAlCaRecoProducers.isoMuonHLT_cfi import *
seqALCARECOHcalCalHO = cms.Sequence(isoMuonHLT*hoCalibProducer)

