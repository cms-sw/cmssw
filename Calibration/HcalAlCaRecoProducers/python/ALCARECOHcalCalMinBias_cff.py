import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------

import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
hcalDigiAlCaMB = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi
hbherecoNoise = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi.hbhereco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
hfrecoNoise = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi.hfreco.clone()
import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi
horecoNoise = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi.horeco.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
hcalminbiasHLT =  HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#    HLTPaths = ['AlCa_HcalPhiSym'],
    eventSetupPathsKey='HcalCalMinBias',
    throw = False #dont throw except on unknown path name 
)

seqALCARECOHcalCalMinBias = cms.Sequence(hcalminbiasHLT*hcalDigiAlCaMB*hbherecoNoise*hfrecoNoise*horecoNoise)

hcalDigiAlCaMB.firstSample = 0
hcalDigiAlCaMB.lastSample = 9
hcalDigiAlCaMB.InputLabel = 'source'
hbherecoNoise.firstSample = 0
hbherecoNoise.samplesToAdd = 4
hbherecoNoise.digiLabel = 'hcalDigiAlCaMB'
hfrecoNoise.firstSample = 1
hfrecoNoise.samplesToAdd = 1
hfrecoNoise.digiLabel = 'hcalDigiAlCaMB'
horecoNoise.firstSample = 0
horecoNoise.samplesToAdd = 4
horecoNoise.digiLabel = 'hcalDigiAlCaMB'

#switch off "ZS in reco":
hbherecoNoise.dropZSmarkedPassed = cms.bool(False)
hfrecoNoise.dropZSmarkedPassed = cms.bool(False)
horecoNoise.dropZSmarkedPassed = cms.bool(False)
