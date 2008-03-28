import FWCore.ParameterSet.Config as cms

from Calibration.EcalAlCaRecoProducers.zeeHLT_cff import *
from ElectroWeakAnalysis.ZReco.zToEE_cfi import *
zeeHLTPath = cms.Path(report*zeeHLT+zToEE)

