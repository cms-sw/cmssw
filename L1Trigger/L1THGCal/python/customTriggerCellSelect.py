import FWCore.ParameterSet.Config as cms


def custom_triggercellselect_supertriggercell(process):
    
    process.hgcalConcentratorProducer.ProcessorParameters.Method = cms.string('superTriggerCellSelect')
    return process


def custom_triggercellselect_threshold(process):

    parameters = process.process.hgcalConcentratorProducer.ProcessorParameters    
    parameters.Method = cms.string('thresholdSelect')
    parameters.MethodNData = cms.uint32(999),
    parameters.MethodMaxCellsInModule = cms.uint32(288),
    parameters.MethodlinLSB = cms.double(100./1024.),
    parameters.MethodadcsaturationBH = adcSaturationBH_MIP,
    parameters.MethodadcnBitsBH = adcNbitsBH,
    parameters.MethodTCThreshold_fC = cms.double(0.),
    parameters.MethodTCThresholdBH_MIP = cms.double(0.),
    parameters.Methodtriggercell_threshold_silicon = cms.double(2.), # MipT
    parameters.Methodtriggercell_threshold_scintillator = cms.double(2.) # MipT

    return process
