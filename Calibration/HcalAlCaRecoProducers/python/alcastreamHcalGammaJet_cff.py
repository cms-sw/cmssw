import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL gammajet:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcagammajet_cfi import *
seqAlcastreamHcalGammaJet = cms.Sequence(GammaJetProd)

