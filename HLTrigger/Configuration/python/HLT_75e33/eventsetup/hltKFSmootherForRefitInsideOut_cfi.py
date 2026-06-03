import FWCore.ParameterSet.Config as cms

def _addProcessKFSmootherForRefitInsideOut(process):
    process.hltKFSmootherForRefitInsideOut = cms.ESProducer('KFTrajectorySmootherESProducer',
                                                            ComponentName = cms.string('hltKFSmootherForRefitInsideOut'),
                                                            Propagator = cms.string('hltSmartPropagatorAnyRK'),
                                                            Updator = cms.string('hltESPKFUpdator'),
                                                            Estimator = cms.string('hltChi2EstimatorForRefit'),
                                                            RecoGeometry = cms.string('GlobalDetLayerGeometry'),
                                                            errorRescaling = cms.double(100),
                                                            minHits = cms.int32(3),
                                                            appendToDataLabel = cms.string(''))

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
modifyConfigurationForKFSmootherForRefitInsideOut_ = mtd_at_hlt.makeProcessModifier(_addProcessKFSmootherForRefitInsideOut)
