import FWCore.ParameterSet.Config as cms

#from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
from RecoJets.JetProducers.ak8PFJetsCS_cfi import ak8PFJetsCS#, ak8PFJetsCSConstituents

#change input tag
pfInput = cms.InputTag('particleFlowTmp')

#constituent subtraction
akCs4PFJets = ak8PFJetsCS.clone( 
    src    = pfInput,
    rParam = cms.double(0.4),
    jetPtMin = cms.double(0.0),
    doAreaFastjet = cms.bool(True),
    GhostArea = cms.double(0.005),
    useConstituentSubtraction = cms.bool(False),
    useConstituentSubtractionHi = cms.bool(True),
    etaMap    = cms.InputTag('hiFJRhoProducer','mapEtaEdges'),
    rho       = cms.InputTag('hiFJRhoProducer','mapToRho'),
    rhom      = cms.InputTag('hiFJRhoProducer','mapToRhoM'),
    csAlpha   = cms.double(1.),
    writeJetsWithConst = cms.bool(True)
    #writeCompound = cms.bool(True)
    )
