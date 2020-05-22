import FWCore.ParameterSet.Config as cms

hiFJRhoFlowModulationProducer = cms.EDProducer(
    'HiFJRhoFlowModulationProducer',
    pfCandSource = cms.InputTag('particleFlow'),
    doJettyExclusion = cms.bool(False),
    doFreePlaneFit = cms.bool(False),
    doEvtPlane = cms.bool(False),
    doFlatTest = cms.bool(False),
    jetTag = cms.InputTag("ak4PFJets"),
    EvtPlane = cms.InputTag("hiEvtPlane"),
    evtPlaneLevel = cms.int32(0)
    )
