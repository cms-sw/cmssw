import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL pedestal:
#------------------------------------------------

import HLTrigger.HLTfilters.hltHighLevel_cfi
hcalCalibPedestalHLT =  HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    eventSetupPathsKey='HcalCalPedestal',
    throw = False #dont throw except on unknown path name 
)

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

qie10Digis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()
qie10Digis.InputLabel = cms.InputTag('hltHcalCalibrationRaw')
qie10Digis.FEDs = cms.untracked.vint32(1132)

import RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi
hbherecoPedestal = RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi.hbheprereco.clone(
    digiLabelQIE8  = cms.InputTag("hcalDigiAlCaPedestal"),
    digiLabelQIE11 = cms.InputTag("hcalDigiAlCaPedestal"),
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

hbherecoPedestal.algorithm.firstSampleShift = -100 # for the very beginning of the TS array

import RecoLocalCalo.HcalRecProducers.hfsimplereco_cfi
hfrecoPedestal = RecoLocalCalo.HcalRecProducers.hfsimplereco_cfi.hfsimplereco.clone()
hfrecoPedestal.digiLabel = cms.InputTag('hcalDigiAlCaPedestal')
hfrecoPedestal.firstSample = cms.int32(0)
hfrecoPedestal.samplesToAdd = cms.int32(2)
hfrecoPedestal.dropZSmarkedPassed = cms.bool(False)

import RecoLocalCalo.HcalRecProducers.hosimplereco_cfi
horecoPedestal = RecoLocalCalo.HcalRecProducers.hosimplereco_cfi.hosimplereco.clone()
horecoPedestal.digiLabel = cms.InputTag('hcalDigiAlCaPedestal')
horecoPedestal.firstSample = cms.int32(0)
horecoPedestal.samplesToAdd = cms.int32(4)
horecoPedestal.dropZSmarkedPassed = cms.bool(False)

seqALCARECOHcalCalPedestal = cms.Sequence(hbherecoPedestal*hfrecoPedestal*horecoPedestal) 

seqALCARECOHcalCalPedestalDigi = cms.Sequence(hcalCalibPedestalHLT*
                                              hcalCalibPedestal*
                                              hcalDigiAlCaPedestal*
                                              qie10Digis*
                                              gtDigisAlCaPedestal)

import RecoLocalCalo.HcalRecProducers.hfprereco_cfi
hfprerecoPedestal = RecoLocalCalo.HcalRecProducers.hfprereco_cfi.hfprereco.clone(
    digiLabel = cms.InputTag("hcalDigiAlCaPedestal"),
    dropZSmarkedPassed = cms.bool(False),
    tsFromDB = cms.bool(False),
    sumAllTimeSlices = cms.bool(False),
    forceSOI = cms.int32(0)
)

import RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi
_phase1_hfrecoPedestal = RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi.hfreco.clone(
    inputLabel = cms.InputTag("hfprerecoPedestal"),
    setNoiseFlags = cms.bool(False),
    algorithm = dict(
        Class = cms.string("HFSimpleTimeCheck"),
        rejectAllFailures = cms.bool(False)
    ),
)

_phase1_seqALCARECOHcalCalPedestal = seqALCARECOHcalCalPedestal.copy()
_phase1_seqALCARECOHcalCalPedestal.insert(0,hfprerecoPedestal)

from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toReplaceWith( seqALCARECOHcalCalPedestal, _phase1_seqALCARECOHcalCalPedestal )
run2_HF_2017.toReplaceWith( hfrecoPedestal, _phase1_hfrecoPedestal )

import RecoLocalCalo.HcalRecProducers.hbheplan1_cfi
hbheplan1Pedestal = RecoLocalCalo.HcalRecProducers.hbheplan1_cfi.hbheplan1.clone(
    hbheInput = cms.InputTag("hbheprerecoPedestal")
)

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( hbherecoPedestal,
    processQIE11 = cms.bool(True),
# temporarily disabled until RecoLocalCalo/HcalRecProducers/python/HBHEPhase1Reconstructor_cfi.py:flagParametersQIE11 is filled
#    setNoiseFlagsQIE11 = cms.bool(True),
)

_plan1_seqALCARECOHcalCalPedestal = _phase1_seqALCARECOHcalCalPedestal.copy()
hbheprerecoPedestal = hbherecoPedestal.clone()
_plan1_seqALCARECOHcalCalPedestal.insert(0,hbheprerecoPedestal)
from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toReplaceWith(hbherecoPedestal, hbheplan1Pedestal)
run2_HEPlan1_2017.toReplaceWith(seqALCARECOHcalCalPedestal, _plan1_seqALCARECOHcalCalPedestal)
