import FWCore.ParameterSet.Config as cms

# Hodoscope Reconstruction
from RecoTBCalo.EcalTBHodoscopeReconstructor.ecal2006TBHodoscopeReconstructor_cfi import *
# TDC Reconstruction
from RecoTBCalo.EcalTBTDCReconstructor.ecal2006TBTDCReconstructor_cfi import *
# uncalibrated rechit producer
from RecoTBCalo.EcalTBRecProducers.ecal2006TBWeightUncalibRecHit_cfi import *
localReco2006_rawData = cms.Sequence(ecal2006TBHodoscopeReconstructor*ecal2006TBTDCReconstructor*ecal2006TBWeightUncalibRecHit)

