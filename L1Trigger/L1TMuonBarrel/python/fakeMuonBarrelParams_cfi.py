import FWCore.ParameterSet.Config as cms

import os

l1tbmtfluts_dir = "L1Trigger/L1TMuon/data/bmtf_luts/"

bmbtfParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonBarrelParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

bmtfParams = cms.ESProducer('L1TMuonBarrelParamsESProducer',
    fwVersion = cms.uint32(1),

    AssLUTPath        = cms.string(os.path.join(l1tbmtfluts_dir, 'LUTs_Ass/')),

    OutOfTime_Filter = cms.bool(False),
    BX_min = cms.int32(-9),
    Extrapolation_Filter = cms.int32(1),
    Open_LUTs = cms.bool(False),
    BX_max = cms.int32(7),
    EtaTrackFinder = cms.bool(True),
    Extrapolation_nbits_PhiB = cms.int32(8),
    Extrapolation_nbits_Phi = cms.int32(8),
    Extrapolation_21 = cms.bool(False),
    PT_Assignment_nbits_PhiB = cms.int32(10),
    PT_Assignment_nbits_Phi = cms.int32(12),
    PHI_Assignment_nbits_Phi = cms.int32(12),
    PHI_Assignment_nbits_PhiB = cms.int32(10),
    OutOfTime_Filter_Window = cms.int32(1)
)
