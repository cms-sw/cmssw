import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL hbhemuon:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcahbhemuon_cfi import *
seqAlcastreamHcalHBHEMuon = cms.Sequence(HBHEMuonProd)
