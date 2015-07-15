import FWCore.ParameterSet.Config as cms

JetResolutionESProducer_AK4PFchs = cms.ESProducer("JetResolutionESProducer",
        label = cms.string('AK4PFchs')
)

JetResolutionESProducer_SF_AK4PFchs = cms.ESProducer("JetResolutionScaleFactorESProducer",
        label = cms.string('AK4PFchs')
)
