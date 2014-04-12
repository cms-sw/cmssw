import FWCore.ParameterSet.Config as cms

# Hodoscope Reconstruction
from RecoTBCalo.EcalTBHodoscopeReconstructor.ecal2004TBHodoscopeReconstructor_cfi import *
# TDC Reconstruction
from RecoTBCalo.EcalTBTDCReconstructor.ecal2004TBTDCReconstructor_cfi import *
# uncalibrated rechit producer
from RecoTBCalo.EcalTBRecProducers.ecal2004TBWeightUncalibRecHit_cfi import *
localReco2004_rawData = cms.Sequence(ecal2004TBHodoscopeReconstructor*ecal2004TBTDCReconstructor*ecal2004TBWeightUncalibRecHit)

