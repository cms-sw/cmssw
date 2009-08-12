import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------

import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
hcalDigiAlCaMB = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi
hbherecoNoise = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi.hbhereco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi
hbherecoMB = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi.hbhereco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
hfrecoNoise = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi.hfreco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
hfrecoMB = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi.hfreco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi
horecoNoise = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi.horeco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi
horecoMB = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi.horeco.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
hcalminbiasHLT =  HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#    HLTPaths = ['AlCa_HcalPhiSym'],
    eventSetupPathsKey='HcalCalMinBias',
    throw = False #dont throw except on unknown path name 
)

seqALCARECOHcalCalMinBias = cms.Sequence(hcalminbiasHLT*hcalDigiAlCaMB*hbherecoNoise*hbherecoMB*hfrecoNoise*hfrecoMB*horecoNoise*horecoMB)

hcalDigiAlCaMB.firstSample = 0
hcalDigiAlCaMB.lastSample = 9
hcalDigiAlCaMB.InputLabel = 'hltAlCaHcalFEDSelector'
hbherecoNoise.firstSample = 0
hbherecoNoise.samplesToAdd = 4
hbherecoNoise.digiLabel = 'hcalDigiAlCaMB'
hbherecoMB.firstSample = 4
hbherecoMB.samplesToAdd = 4
hbherecoMB.digiLabel = 'hcalDigiAlCaMB'
hfrecoNoise.firstSample = 1
hfrecoNoise.samplesToAdd = 1
hfrecoNoise.digiLabel = 'hcalDigiAlCaMB'
hfrecoMB.firstSample = 3
hfrecoMB.samplesToAdd = 1
hfrecoMB.digiLabel = 'hcalDigiAlCaMB'
horecoNoise.firstSample = 0
horecoNoise.samplesToAdd = 4
horecoNoise.digiLabel = 'hcalDigiAlCaMB'
horecoMB.firstSample = 4
horecoMB.samplesToAdd = 4
horecoMB.digiLabel = 'hcalDigiAlCaMB'

#switch off "ZS in reco":
hbherecoNoise.dropZSmarkedPassed = cms.bool(False)
hfrecoNoise.dropZSmarkedPassed = cms.bool(False)
hbherecoMB.dropZSmarkedPassed = cms.bool(False)
hfrecoMB.dropZSmarkedPassed = cms.bool(False)

