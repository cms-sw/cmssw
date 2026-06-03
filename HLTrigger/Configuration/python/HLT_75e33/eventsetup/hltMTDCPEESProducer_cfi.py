import FWCore.ParameterSet.Config as cms

def _addProcessMTDCPEESProducer(process):
    process.hltMTDCPEESProducer = cms.ESProducer('MTDCPEESProducer',
                                                 appendToDataLabel = cms.string(''))

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
modifyConfigurationForMTDCPEESProducer_ = mtd_at_hlt.makeProcessModifier(_addProcessMTDCPEESProducer)
