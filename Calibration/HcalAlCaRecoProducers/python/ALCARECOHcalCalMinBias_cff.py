import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------

import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
hcalDigiAlCaMB = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi
hbherecoMBspecial = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi.hbheprereco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi
horecoMBspecial = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi.horeco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi
hbherecoNoise = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi.hbheprereco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
hfrecoNoise = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi.hfreco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
hfrecoMBspecial = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi.hfreco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi
horecoNoise = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi.horeco.clone()

#add GT digi:
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
gtDigisAlCaMB = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()

import HLTrigger.HLTfilters.hltHighLevel_cfi
hcalminbiasHLT =  HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#    HLTPaths = ['HLT_HcalPhiSym'],
    eventSetupPathsKey='HcalCalMinBias',
    throw = False #dont throw except on unknown path name 
)

seqALCARECOHcalCalMinBias = cms.Sequence(hcalminbiasHLT*hcalDigiAlCaMB*gtDigisAlCaMB*hbherecoMBspecial*horecoMBspecial*hbherecoNoise*hfrecoNoise*hfrecoMBspecial*horecoNoise)
seqALCARECOHcalCalMinBiasNoHLT = cms.Sequence(hcalDigiAlCaMB*gtDigisAlCaMB*hbherecoMBspecial*horecoMBspecial*hbherecoNoise*hfrecoNoise*hfrecoMBspecial*horecoNoise)

gtDigisAlCaMB.DaqGtInputTag = 'source'

hcalDigiAlCaMB.firstSample = 0
hcalDigiAlCaMB.lastSample = 9
hcalDigiAlCaMB.InputLabel = 'rawDataCollector'

hbherecoMBspecial.firstSample = 0
hbherecoMBspecial.samplesToAdd = 4
hbherecoMBspecial.digiLabel = 'hcalDigiAlCaMB'

horecoMBspecial.firstSample = 0
horecoMBspecial.samplesToAdd = 4
horecoMBspecial.digiLabel = 'hcalDigiAlCaMB'


hbherecoNoise.firstSample = 0
hbherecoNoise.samplesToAdd = 4
hbherecoNoise.digiLabel = 'hcalDigiAlCaMB'

hfrecoNoise.firstSample = 0
hfrecoNoise.samplesToAdd = 2
hfrecoNoise.digiLabel = 'hcalDigiAlCaMB'

hfrecoMBspecial.firstSample = 2
hfrecoMBspecial.samplesToAdd = 2
hfrecoMBspecial.digiLabel = 'hcalDigiAlCaMB'

horecoNoise.firstSample = 0
horecoNoise.samplesToAdd = 4
horecoNoise.digiLabel = 'hcalDigiAlCaMB'

#switch off "ZS in reco":
hbherecoMBspecial.dropZSmarkedPassed = cms.bool(False)
horecoMBspecial.dropZSmarkedPassed = cms.bool(False)
hbherecoNoise.dropZSmarkedPassed = cms.bool(False)
hfrecoNoise.dropZSmarkedPassed = cms.bool(False)
horecoNoise.dropZSmarkedPassed = cms.bool(False)
hfrecoMBspecial.dropZSmarkedPassed = cms.bool(False)
hbherecoMBspecial.tsFromDB = cms.bool(False)
horecoMBspecial.tsFromDB = cms.bool(False)
hbherecoNoise.tsFromDB = cms.bool(False)
hfrecoNoise.tsFromDB = cms.bool(False)
hfrecoMBspecial.tsFromDB = cms.bool(False)
horecoNoise.tsFromDB = cms.bool(False)

