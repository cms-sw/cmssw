import FWCore.ParameterSet.Config as cms

def _addProcessMkFitGeometry(process):
    process.mkFitGeometryESProducer = cms.ESProducer("MkFitGeometryESProducer",
        appendToDataLabel = cms.string('')
    )

from Configuration.ProcessModifiers.trackingMkFitCommon_cff import trackingMkFitCommon
modifyConfigurationForTrackingMkFitGeometryMkfit_ = trackingMkFitCommon.makeProcessModifier(_addProcessMkFitGeometry)

def _addProcesshltInitialStepMkFitConfig(process):
    process.hltInitialStepTrackCandidatesMkFitConfig = cms.ESProducer("MkFitIterationConfigESProducer",
        ComponentName = cms.string('hltInitialStepTrackCandidatesMkFitConfig'),
        appendToDataLabel = cms.string(''),
        config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-initialStep.json'),
        maxClusterSize = cms.uint32(8),
        minPt = cms.double(0.8)
    )

def _addProcesshltLSTStepMkFitConfig(process):
    process.hltInitialStepTrackCandidatesMkFitConfig = cms.ESProducer("MkFitIterationConfigESProducer",
        ComponentName = cms.string('hltInitialStepTrackCandidatesMkFitConfig'),
        appendToDataLabel = cms.string(''),
        config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-lstStep.json'),
        maxClusterSize = cms.uint32(8),
        minPt = cms.double(0.9)
    )

from Configuration.ProcessModifiers.hltTrackingMkFitInitialStep_cff import hltTrackingMkFitInitialStep
from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
modifyConfigurationForTrackingMkFithltInitialStepMkFitConfig_ = (~seedingLST & hltTrackingMkFitInitialStep).makeProcessModifier(_addProcesshltInitialStepMkFitConfig)
modifyConfigurationForTrackingMkFithltLSTStepMkFitConfig_ = (seedingLST & hltTrackingMkFitInitialStep).makeProcessModifier(_addProcesshltLSTStepMkFitConfig)
