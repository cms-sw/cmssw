import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL gammajet:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcagammajet_cfi import *
from Calibration.HcalAlCaRecoProducers.gammajetHLT_cfi import *
seqALCARECOHcalCalGammaJet = cms.Sequence(gammajetHLT*GammaJetProd)

