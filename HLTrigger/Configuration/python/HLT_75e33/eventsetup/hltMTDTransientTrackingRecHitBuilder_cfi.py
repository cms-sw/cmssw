import FWCore.ParameterSet.Config as cms

def _addProcessMTDTransientTrackingRecHitBuilder(process):
    process.hltMTDTransientTrackingRecHitBuilder = cms.ESProducer("MTDTransientTrackingRecHitBuilderESProducer",
                                                                  ComponentName = cms.string('hltMTDRecHitBuilder'))

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
modifyConfigurationForMTDTransientTrackingRecHitBuilder_ = mtd_at_hlt.makeProcessModifier(_addProcessMTDTransientTrackingRecHitBuilder)
