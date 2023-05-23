import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonsHgcalLCIsodR0p2dRVetoEM0p00dRVetoHad0p02minEEM0p00minEHad0p00 = cms.EDProducer("MuonHLTHGCalLayerClusterIsolationProducer",
    doRhoCorrection = cms.bool(False),
    drMax = cms.double(0.2),
    drVetoEM = cms.double(0.0),
    drVetoHad = cms.double(0.02),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    layerClusterProducer = cms.InputTag("hgcalLayerClusters"),
    minEnergyEM = cms.double(0.0),
    minEnergyHad = cms.double(0.0),
    minEtEM = cms.double(0.0),
    minEtHad = cms.double(0.0),
    recoCandidateProducer = cms.InputTag("hltPhase2L3MuonCandidates"),
    rhoMax = cms.double(99999999.0),
    rhoProducer = cms.InputTag("hltFixedGridRhoFastjetAllCaloForEGamma"),
    rhoScale = cms.double(1.0),
    useEt = cms.bool(False)
)
