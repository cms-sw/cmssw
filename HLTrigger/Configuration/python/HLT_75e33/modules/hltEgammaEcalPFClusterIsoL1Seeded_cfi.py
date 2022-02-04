import FWCore.ParameterSet.Config as cms

hltEgammaEcalPFClusterIsoL1Seeded = cms.EDProducer("EgammaHLTEcalPFClusterIsolationProducer",
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    doRhoCorrection = cms.bool(False),
    drMax = cms.double(0.2),
    drVetoBarrel = cms.double(0.0),
    drVetoEndcap = cms.double(0.0),
    effectiveAreas = cms.vdouble(0.29, 0.21),
    energyBarrel = cms.double(0.0),
    energyEndcap = cms.double(0.0),
    etaStripBarrel = cms.double(0.0),
    etaStripEndcap = cms.double(0.0),
    pfClusterProducer = cms.InputTag("hltParticleFlowClusterECALL1Seeded"),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    rhoMax = cms.double(99999999.0),
    rhoProducer = cms.InputTag("hltFixedGridRhoFastjetAllCaloForEGamma"),
    rhoScale = cms.double(1.0)
)
