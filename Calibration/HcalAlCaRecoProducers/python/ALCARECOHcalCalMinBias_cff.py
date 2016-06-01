import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------

import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
hcalDigiAlCaMB = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi
hbherecoMBNZS = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi.hbheprereco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi
horecoMBNZS = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi.horeco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi
hbherecoNoise = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi.hbheprereco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
hfrecoNoise = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi.hfreco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
hfrecoMBNZS = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi.hfreco.clone()
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

seqALCARECOHcalCalMinBias = cms.Sequence(hcalminbiasHLT*hcalDigiAlCaMB*gtDigisAlCaMB*hbherecoMBNZS*horecoMBNZS*hbherecoNoise*hfrecoNoise*hfrecoMBNZS*horecoNoise)
seqALCARECOHcalCalMinBiasNoHLT = cms.Sequence(hcalDigiAlCaMB*gtDigisAlCaMB*hbherecoMBNZS*horecoMBNZS*hbherecoNoise*hfrecoNoise*hfrecoMBNZS*horecoNoise)

gtDigisAlCaMB.DaqGtInputTag = 'source'

hcalDigiAlCaMB.firstSample = 0
hcalDigiAlCaMB.lastSample = 9
hcalDigiAlCaMB.InputLabel = 'rawDataCollector'

hbherecoMBNZS.firstSample = 0
hbherecoMBNZS.samplesToAdd = 4
hbherecoMBNZS.digiLabel = 'hcalDigiAlCaMB'

horecoMBNZS.firstSample = 0
horecoMBNZS.samplesToAdd = 4
horecoMBNZS.digiLabel = 'hcalDigiAlCaMB'


hbherecoNoise.firstSample = 0
hbherecoNoise.samplesToAdd = 4
hbherecoNoise.digiLabel = 'hcalDigiAlCaMB'

hfrecoNoise.firstSample = 0
hfrecoNoise.samplesToAdd = 2
hfrecoNoise.digiLabel = 'hcalDigiAlCaMB'

hfrecoMBNZS.firstSample = 2
hfrecoMBNZS.samplesToAdd = 2
hfrecoMBNZS.digiLabel = 'hcalDigiAlCaMB'

horecoNoise.firstSample = 0
horecoNoise.samplesToAdd = 4
horecoNoise.digiLabel = 'hcalDigiAlCaMB'

#switch off "ZS in reco":
hbherecoMBNZS.dropZSmarkedPassed = cms.bool(False)
horecoMBNZS.dropZSmarkedPassed = cms.bool(False)
hbherecoNoise.dropZSmarkedPassed = cms.bool(False)
hfrecoNoise.dropZSmarkedPassed = cms.bool(False)
horecoNoise.dropZSmarkedPassed = cms.bool(False)
hfrecoMBNZS.dropZSmarkedPassed = cms.bool(False)
hbherecoMBNZS.tsFromDB = cms.bool(False)
horecoMBNZS.tsFromDB = cms.bool(False)
hbherecoNoise.tsFromDB = cms.bool(False)
hfrecoNoise.tsFromDB = cms.bool(False)
hfrecoMBNZS.tsFromDB = cms.bool(False)
horecoNoise.tsFromDB = cms.bool(False)

