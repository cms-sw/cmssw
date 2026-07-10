import FWCore.ParameterSet.Config as cms

def _addProcessKFFitterForRefitInsideOut(process):
    process.hltKFFitterForRefitInsideOut = cms.ESProducer('KFTrajectoryFitterESProducer',
                                                          RecoGeometry = cms.string('GlobalDetLayerGeometry'),
                                                          ComponentName = cms.string('hltKFFitterForRefitInsideOut'),
                                                          Propagator = cms.string('hltSmartPropagatorAnyRK'),
                                                          Updator = cms.string('hltESPKFUpdator'),
                                                          Estimator = cms.string('hltChi2EstimatorForRefit'),
                                                          minHits = cms.int32(3),
                                                          appendToDataLabel = cms.string(''))

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
modifyConfigurationForKFFitterForRefitInsideOut_ = mtd_at_hlt.makeProcessModifier(_addProcessKFFitterForRefitInsideOut)
