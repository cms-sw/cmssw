import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector

goodOfflinePrimaryVertices = cms.EDFilter("PrimaryVertexObjectFilter",
    filterParams = pvSelector.clone(
        minNdof = cms.double(4.0),
        maxZ = cms.double(24.0)
    ),
    src = cms.InputTag('offlinePrimaryVertices')
)

from CommonTools.ParticleFlow.pfPileUp_cfi import pfPileUp
pfPileUpForAK5PFchsJets = pfPileUp.clone(
    PFCandidates = cms.InputTag('particleFlow'),
    Vertices = cms.InputTag('goodOfflinePrimaryVertices'),
    checkClosestZVertex = cms.bool(False)
)
from CommonTools.ParticleFlow.TopProjectors.pfNoPileUp_cfi import pfNoPileUp
pfNoPileUpForAK5PFchsJets = pfNoPileUp.clone(
    topCollection = cms.InputTag('pfPileUpForAK5PFchsJets'),
    bottomCollection = cms.InputTag('particleFlow')
)

from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets
ak5PFchsJets = ak5PFJets.clone(
    src = cms.InputTag('pfNoPileUpForAK5PFchsJets'),
    doAreaFastjet = cms.bool(True)
)

from CommonTools.ParticleFlow.ParticleSelectors.pfAllNeutralHadrons_cfi import pfAllNeutralHadrons
from CommonTools.ParticleFlow.ParticleSelectors.pfAllPhotons_cfi import pfAllPhotons
pfNeutralCandPdgIds = []
pfNeutralCandPdgIds.extend(pfAllNeutralHadrons.pdgId.value())
pfNeutralCandPdgIds.extend(pfAllPhotons.pdgId.value()) 
pfNeutralCandsForAK5PFchsJets = cms.EDFilter("PdgIdPFCandidateSelector",
    src = cms.InputTag('particleFlow'),
    pdgId = cms.vint32(pfNeutralCandPdgIds)
)

from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets
kt6PFchsJets = kt4PFJets.clone(
    ##src = cms.InputTag('pfNeutralCandsForAK5PFchsJets'),
    src = cms.InputTag('particleFlow'), 
    doAreaFastjet = cms.bool(True),
    doRhoFastjet = cms.bool(True),
    rParam = cms.double(0.6)
)

ak5PFchsJetsSequence = cms.Sequence(
    goodOfflinePrimaryVertices
   + pfPileUpForAK5PFchsJets
   + pfNoPileUpForAK5PFchsJets
   + ak5PFchsJets
   + pfNeutralCandsForAK5PFchsJets
   + kt6PFchsJets
)    
