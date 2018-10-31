import FWCore.ParameterSet.Config as cms


def custom_triggercellselect_supertriggercell(process):
    
    process.hgcalConcentratorProducer.ProcessorParameters.Method = cms.string('superTriggerCellSelect')
    return process


def custom_triggercellselect_threshold(process):
    
    process.hgcalConcentratorProducer.ProcessorParameters.Method = cms.string('thresholdSelect')
    return process
