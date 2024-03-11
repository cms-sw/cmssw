import FWCore.ParameterSet.Config as cms

# Hodoscope Reconstruction
from RecoTBCalo.EcalTBHodoscopeReconstructor.ecal2006TBHodoscopeReconstructor_cfi import *
# TDC Reconstruction
from RecoTBCalo.EcalTBTDCReconstructor.ecal2007H4TBTDCReconstructor_cfi import *
# uncalibrated rechit producer
from RecoTBCalo.EcalTBRecProducers.ecal2007TBWeightUncalibRecHit_cfi import *
localReco2007h4_rawData = cms.Sequence(ecal2006TBHodoscopeReconstructor*ecal2007H4TBTDCReconstructor*ecal2007TBWeightUncalibRecHit)
# foo bar baz
