import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL HO:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcahomuoncosmics_cfi import *
from Calibration.HcalAlCaRecoProducers.isoMuonHLT_cfi import *
seqALCARECOHcalCalHOCosmics = cms.Sequence(isoMuonHLT*hoCalibCosmicsProducer)
#seqALCARECOHcalCalHOCosmics = cms.Sequence(hoCalibProducer)


