import FWCore.ParameterSet.Config as cms

# ECAL conditions
from CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetrieverTB_cfi import *
from CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi import *
# Hodoscope Reconstruction
from RecoTBCalo.EcalTBHodoscopeReconstructor.ecalTBSimHodoscopeReconstructor_cfi import *
# TDC Reconstruction
from RecoTBCalo.EcalTBTDCReconstructor.ecalTBSimTDCReconstructor_cfi import *
# ECAL reconstruction
from RecoTBCalo.EcalTBRecProducers.ecalTBSimWeightUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalTBSimRecHit_cfi import *
localReco_tbsim = cms.Sequence(ecalTBSimHodoscopeReconstructor*ecalTBSimTDCReconstructor*ecalTBSimWeightUncalibRecHit*ecalTBSimRecHit)
from RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi import *
