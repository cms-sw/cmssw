import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL pedestal:
#------------------------------------------------

import EventFilter.HcalRawToDigi.HcalCalibTypeFilter_cfi
hcalCalibPedestal = EventFilter.HcalRawToDigi.HcalCalibTypeFilter_cfi.hcalCalibTypeFilter.clone(
    #  InputLabel = cms.string('rawDataCollector'),
    InputLabel = cms.string('hltHcalCalibrationRaw::HLT'),
    #  InputLabel = cms.InputTag("hltEcalCalibrationRaw","","HLT"),
    CalibTypes    = cms.vint32( 1 ),
    FilterSummary = cms.untracked.bool( False )
    )

#add GT digi:
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
gtDigisAlCaPedestal = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()

import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
hcalDigiAlCaPedestal = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()
hcalDigiAlCaPedestal.InputLabel = cms.InputTag('hltHcalCalibrationRaw')

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi
hbherecoPedestal = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi.hbheprereco.clone()
hbherecoPedestal.digiLabel = cms.InputTag('hcalDigiAlCaPedestal')
hbherecoPedestal.firstSample = cms.int32(0)
hbherecoPedestal.samplesToAdd = cms.int32(4)

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
hfrecoPedestal = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi.hfreco.clone()
hfrecoPedestal.digiLabel = cms.InputTag('hcalDigiAlCaPedestal')
hfrecoPedestal.firstSample = cms.int32(0)
hfrecoPedestal.samplesToAdd = cms.int32(2)

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi
horecoPedestal = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi.horeco.clone()
horecoPedestal.digiLabel = cms.InputTag('hcalDigiAlCaPedestal')
horecoPedestal.firstSample = cms.int32(0)
horecoPedestal.samplesToAdd = cms.int32(4)

# switch off "Hcal ZS in reco":
hbherecoPedestal.dropZSmarkedPassed = cms.bool(False)
hfrecoPedestal.dropZSmarkedPassed = cms.bool(False)
horecoPedestal.dropZSmarkedPassed = cms.bool(False)

hcalLocalRecoSequencePedestal = cms.Sequence(hbherecoPedestal*hfrecoPedestal*horecoPedestal) 

seqALCARECOHcalCalPedestal = cms.Sequence(hcalCalibPedestal*
                                          hcalDigiAlCaPedestal*
                                          gtDigisAlCaPedestal*
                                          hcalLocalRecoSequencePedestal)
