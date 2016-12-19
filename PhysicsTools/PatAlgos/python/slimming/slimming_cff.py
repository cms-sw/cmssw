import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.packedPFCandidates_cff import *
from PhysicsTools.PatAlgos.slimming.lostTracks_cfi import *
from PhysicsTools.PatAlgos.slimming.offlineSlimmedPrimaryVertices_cfi import *
from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import *
from PhysicsTools.PatAlgos.slimming.genParticles_cff import *
from PhysicsTools.PatAlgos.slimming.selectedPatTrigger_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedJets_cfi      import *
from PhysicsTools.PatAlgos.slimming.slimmedGenJets_cfi   import *
from PhysicsTools.PatAlgos.slimming.slimmedElectrons_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedMuons_cfi     import *
from PhysicsTools.PatAlgos.slimming.slimmedPhotons_cfi   import *
from PhysicsTools.PatAlgos.slimming.slimmedTaus_cfi      import *
from PhysicsTools.PatAlgos.slimming.slimmedSecondaryVertices_cfi      import *
from PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi      import *
from PhysicsTools.PatAlgos.slimming.metFilterPaths_cff   import *
from PhysicsTools.PatAlgos.slimming.MicroEventContent_cff import *
from RecoEgamma.EgammaPhotonProducers.reducedEgamma_cfi  import *
from RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi import bunchSpacingProducer

slimmingTask = cms.Task(
    packedPFCandidatesTask,
    lostTracks,
    offlineSlimmedPrimaryVertices,
    primaryVertexAssociation,
    genParticlesTask,
    selectedPatTrigger,
    slimmedJets,
    slimmedJetsAK8,
    slimmedGenJets,
    slimmedGenJetsAK8,
    slimmedElectrons,
    slimmedMuons,
    slimmedPhotons,
    slimmedTaus,
    slimmedSecondaryVertices,
    slimmedMETs,
    metFilterPathsTask,
    reducedEgamma,
    bunchSpacingProducer
)
