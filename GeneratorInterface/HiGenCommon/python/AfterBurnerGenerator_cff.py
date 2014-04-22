import FWCore.ParameterSet.Config as cms

AftBurner = cms.EDProducer("AfterBurnerGenerator",
                           src = cms.InputTag("generator"),
                           modv1= cms.InputTag("0.0"),
                           modv2= cms.InputTag("0.2"),
                           modv3= cms.InputTag("0.0"),
                           modv4= cms.InputTag("0.0"),
                           modv5= cms.InputTag("0.0"),
                           modv6= cms.InputTag("0.0"),
                           fluct_v1= cms.double(0.0),
                           fluct_v2= cms.double(0.01),
                           fluct_v3= cms.double(0.0),
                           fluct_v4= cms.double(0.0),
                           fluct_v5= cms.double(0.0),
                           fluct_v6= cms.double(0.0),
                           fixEP = cms.untracked.bool(True)
)                           

