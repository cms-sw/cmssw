import FWCore.ParameterSet.Config as cms

# configuration for HLTBtagLifetimeAnalyzer
hltBtagLifetimeAnalyzer = cms.EDAnalyzer("HLTBtagLifetimeAnalyzer",
    mcRadius = cms.double(0.1),
    outputFile = cms.string('plots.root'),
    offlineRadius = cms.double(0.1),
    vertex = cms.InputTag("pixelVertices"),
    triggerEvent = cms.InputTag("triggerSummaryRAW","","HLT"),
    offlineBJets = cms.InputTag("jetProbabilityBJetTags"),
    triggerPath = cms.string('HLTB1Jet'),
    levels = cms.VPSet(cms.PSet(
        filter = cms.InputTag("hltBLifetimeL1seeds"),
        jets = cms.InputTag("iterativeCone5CaloJets","","HLT"),
        name = cms.string('preL2'),
        title = cms.string('pre-L2')
    ), cms.PSet(
        filter = cms.InputTag("hltBLifetime1jetL2filter"),
        jets = cms.InputTag("hltBLifetimeL25Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25Associator","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter"),
        jets = cms.InputTag("hltBLifetimeL3Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3Associator","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter"),
        jets = cms.InputTag("hltBLifetimeHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    )),
    offlineCuts = cms.PSet(
        cut50 = cms.double(0.6),
        cut20 = cms.double(1.1),
        cut80 = cms.double(0.3)
    ),
    jetConfiguration = cms.PSet(
        maxEta = cms.double(5.0),
        maxEnergy = cms.double(300.0)
    ),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    mcFlavours = cms.PSet(
        light = cms.vuint32(1, 2, 3, 21),
        c = cms.vuint32(4),
        b = cms.vuint32(5),
        g = cms.vuint32(21),
        uds = cms.vuint32(1, 2, 3)
    ),
    mcPartons = cms.InputTag("hltIC5byValAlgo"),
    vertexConfiguration = cms.PSet(
        maxZ = cms.double(20.0),
        maxR = cms.double(0.05)
    )
)


