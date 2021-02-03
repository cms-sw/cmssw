import FWCore.ParameterSet.Config as cms

convStepChi2Est = cms.ESProducer("Chi2ChargeMeasurementEstimatorESProducer",
    ComponentName = cms.string('convStepChi2Est'),
    MaxChi2 = cms.double(30.0),
    MaxDisplacement = cms.double(100),
    MaxSagitta = cms.double(-1.0),
    MinPtForHitRecoveryInGluedDet = cms.double(1000000000000),
    MinimalTolerance = cms.double(0.5),
    appendToDataLabel = cms.string(''),
    clusterChargeCut = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutTight')
    ),
    nSigma = cms.double(3.0),
    pTChargeCutThreshold = cms.double(-1)
)
