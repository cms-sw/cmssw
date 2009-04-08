def initL1RSSubsystems( tagBase = 'IDEAL',
                        objectKey = 'dummy' ):

    import FWCore.ParameterSet.Config as cms

    initL1RSSubsystems.params = cms.PSet(
        recordInfo = cms.VPSet(
        cms.PSet(
            record = cms.string('L1MuGMTChannelMaskRcd'),
            tag = cms.string('L1MuGMTChannelMask_' + tagBase),
            type = cms.string('L1MuGMTChannelMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1RCTChannelMaskRcd'),
            tag = cms.string('L1RCTChannelMask_' + tagBase),
            type = cms.string('L1RCTChannelMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GctChannelMaskRcd'),
            tag = cms.string('L1GctChannelMask_' + tagBase),
            type = cms.string('L1GctChannelMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtPrescaleFactorsAlgoTrigRcd'),
            tag = cms.string('L1GtPrescaleFactorsAlgoTrig_' + tagBase),
            type = cms.string('L1GtPrescaleFactors'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtPrescaleFactorsTechTrigRcd'),
            tag = cms.string('L1GtPrescaleFactorsTechTrig_' + tagBase),
            type = cms.string('L1GtPrescaleFactors'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskAlgoTrigRcd'),
            tag = cms.string('L1GtTriggerMaskAlgoTrig_' + tagBase),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskTechTrigRcd'),
            tag = cms.string('L1GtTriggerMaskTechTrig_' + tagBase),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskVetoTechTrigRcd'),
            tag = cms.string('L1GtTriggerMaskVetoTechTrig_' + tagBase),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(objectKey)
        ))
        )
