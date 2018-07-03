import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------

import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
hcalDigiAlCaMB = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()

hcalDigiAlCaMB.firstSample = 0
hcalDigiAlCaMB.lastSample = 9
hcalDigiAlCaMB.InputLabel = 'rawDataCollector'

#add GT digi:
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
gtDigisAlCaMB = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()

gtDigisAlCaMB.DaqGtInputTag = 'source'

import RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi
hbherecoNoise = RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi.hbheprereco.clone(
    digiLabelQIE8  = cms.InputTag("hcalDigiAlCaMB"),
    digiLabelQIE11 = cms.InputTag("hcalDigiAlCaMB"),
###    tsFromDB = cms.bool(False),
    dropZSmarkedPassed = cms.bool(False),
    algorithm = dict(
        useMahi = cms.bool(False),
        useM2 = cms.bool(False),
        useM3 = cms.bool(False)
    ),
    processQIE11 = cms.bool(False),
    setNegativeFlagsQIE8 = cms.bool(False),
    setNegativeFlagsQIE11 = cms.bool(False),
    setNoiseFlagsQIE8 = cms.bool(True),
    setNoiseFlagsQIE11 = cms.bool(False),
    setPulseShapeFlagsQIE8 = cms.bool(False),
    setPulseShapeFlagsQIE11 = cms.bool(False),
    setLegacyFlagsQIE8 = cms.bool(False),
    setLegacyFlagsQIE11 = cms.bool(False),
)

hbherecoNoise.algorithm.firstSampleShift = -100 # to set reco window at very beginning of the TS array

import RecoLocalCalo.HcalRecProducers.hosimplereco_cfi
horecoNoise = RecoLocalCalo.HcalRecProducers.hosimplereco_cfi.hosimplereco.clone()

horecoNoise.firstSample = 0
horecoNoise.samplesToAdd = 4
horecoNoise.digiLabel = 'hcalDigiAlCaMB'
horecoNoise.tsFromDB = cms.bool(False)
horecoNoise.dropZSmarkedPassed = cms.bool(False)

import RecoLocalCalo.HcalRecProducers.hfsimplereco_cfi
hfrecoNoise = RecoLocalCalo.HcalRecProducers.hfsimplereco_cfi.hfsimplereco.clone()

hfrecoNoise.firstSample = 0
hfrecoNoise.samplesToAdd = 2
hfrecoNoise.digiLabel = 'hcalDigiAlCaMB'
hfrecoNoise.tsFromDB = cms.bool(False)
hfrecoNoise.dropZSmarkedPassed = cms.bool(False)


