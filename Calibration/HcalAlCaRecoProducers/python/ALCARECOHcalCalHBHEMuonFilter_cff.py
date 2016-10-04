import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
#AlCaReco filtering for HCAL HBHEMuon:
#-------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaHBHEMuonFilter_cfi import *

seqALCARECOHcalCalHBHEMuonFilter = cms.Sequence(AlcaHBHEMuonFilter)
