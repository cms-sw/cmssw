import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------
import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
hcalDigiAlCaPedestals = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()
hcalDigiAlCaPedestals.InputLabel = 'hltHcalCalibrationRaw'

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi
hbherecoPedestals = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi.hbheprereco.clone()

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
hfrecoPedestals = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi.hfreco.clone()

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
horecoPedestals = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi.horeco.clone()

#add GT digi:
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
gtDigisAlCaPedestals = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()

hbherecoPedestals.firstSample = 0
hbherecoPedestals.samplesToAdd = 4
hbherecoPedestals.digiLabel = 'hcalDigiAlCaPedestals'

hfrecoPedestals.firstSample = 0
hfrecoPedestals.samplesToAdd = 2
hfrecoPedestals.digiLabel = 'hcalDigiAlCaPedestals'

horecoPedestals.firstSample = 0
horecoPedestals.samplesToAdd = 4
horecoPedestals.digiLabel = 'hcalDigiAlCaPedestals'

# switch off "Hcal ZS in reco":
hbherecoPedestals.dropZSmarkedPassed = cms.bool(False)
hfrecoPedestals.dropZSmarkedPassed = cms.bool(False)
horecoPedestals.dropZSmarkedPassed = cms.bool(False)

hcalLocalRecoSequencePedestals = cms.Sequence(hbherecoPedestals*hfrecoPedestals*horecoPedestals) 

import EventFilter.HcalRawToDigi.HcalCalibTypeFilter_cfi
hcalCalibPedestal = EventFilter.HcalRawToDigi.HcalCalibTypeFilter_cfi.hcalCalibTypeFilter.clone(
    #  InputLabel = cms.string('rawDataCollector'), 
    InputLabel = cms.string('hltHcalCalibrationRaw::HLT'),
    #  InputLabel = cms.InputTag("hltEcalCalibrationRaw","","HLT"),
    CalibTypes    = cms.vint32( 1 ),
    FilterSummary = cms.untracked.bool( False )
    )

seqALCARECOHcalCalPedestal = cms.Sequence(hcalCalibPedestal*
                                          hcalDigiAlCaPedestals*
                                          gtDigisAlCaPedestals*
                                          hcalLocalRecoSequencePedestals*
                                          hbherecoPedestals*
                                          hfrecoPedestals*
                                          horecoPedestals)
