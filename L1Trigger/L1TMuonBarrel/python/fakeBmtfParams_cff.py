import FWCore.ParameterSet.Config as cms

import os

l1tbmtfluts_dir = "L1Trigger/L1TMuon/data/bmtf_luts/"
#each bit of the mask corresponds to one sector
maskenable      = '000000000000'
maskdisable     = '111111111111'

bmbtfParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonBarrelParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeBmtfParams = cms.ESProducer('L1TMuonBarrelParamsESProducer',
    fwVersion = cms.uint32(2),

    AssLUTPath        = cms.string(os.path.join(l1tbmtfluts_dir, 'LUTs_Ass/')),

    OutOfTime_Filter = cms.bool(False),
    BX_min = cms.int32(-2),
    Extrapolation_Filter = cms.int32(1),
    Open_LUTs = cms.bool(False),
    BX_max = cms.int32(2),
    EtaTrackFinder = cms.bool(True),
    Extrapolation_nbits_PhiB = cms.int32(8),
    Extrapolation_nbits_Phi = cms.int32(8),
    Extrapolation_21 = cms.bool(False),
    PT_Assignment_nbits_PhiB = cms.int32(10),
    PT_Assignment_nbits_Phi = cms.int32(12),
    PHI_Assignment_nbits_Phi = cms.int32(12),
    PHI_Assignment_nbits_PhiB = cms.int32(10),
    OutOfTime_Filter_Window = cms.int32(1),
    
    #Each element in vstring corresponds to one TF (-2,-1,-0,+0,+1,+2)
    mask_phtf_st1        = cms.vstring(maskdisable, maskenable, maskenable, maskenable, maskenable, maskenable, maskdisable),
    mask_phtf_st2        = cms.vstring(maskenable,  maskenable, maskenable, maskenable, maskenable, maskenable, maskenable),
    mask_phtf_st3        = cms.vstring(maskenable,  maskenable, maskenable, maskenable, maskenable, maskenable, maskenable),
    mask_phtf_st4        = cms.vstring(maskenable,  maskenable, maskenable, maskenable, maskenable, maskenable, maskenable),

    mask_ettf_st1        = cms.vstring(maskdisable, maskenable, maskenable, maskenable, maskenable, maskenable, maskdisable),
    mask_ettf_st2        = cms.vstring(maskenable,  maskenable, maskenable, maskenable, maskenable, maskenable, maskenable),
    mask_ettf_st3        = cms.vstring(maskenable,  maskenable, maskenable, maskenable, maskenable, maskenable, maskenable)

)
