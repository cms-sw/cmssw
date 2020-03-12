import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.packedPFCandidates_cff import *
from PhysicsTools.PatAlgos.slimming.isolatedTracks_cfi import *
from PhysicsTools.PatAlgos.slimming.lostTracks_cfi import *
from PhysicsTools.PatAlgos.slimming.offlineSlimmedPrimaryVertices_cfi import *
from PhysicsTools.PatAlgos.slimming.offlineSlimmedPrimaryVertices4D_cfi import *
from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import *
from PhysicsTools.PatAlgos.slimming.genParticles_cff import *
from PhysicsTools.PatAlgos.slimming.selectedPatTrigger_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedPatTrigger_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedJets_cfi      import *
from PhysicsTools.PatAlgos.slimming.slimmedCaloJets_cfi  import *
from PhysicsTools.PatAlgos.slimming.slimmedGenJets_cfi   import *
from PhysicsTools.PatAlgos.slimming.slimmedElectrons_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedLowPtElectrons_cfi import *
from PhysicsTools.PatAlgos.slimming.lowPtGsfLinks_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedMuons_cfi     import *
from PhysicsTools.PatAlgos.slimming.slimmedPhotons_cfi   import *
from PhysicsTools.PatAlgos.slimming.slimmedOOTPhotons_cff import *
from PhysicsTools.PatAlgos.slimming.slimmedTaus_cfi      import *
from PhysicsTools.PatAlgos.slimming.slimmedSecondaryVertices_cfi      import *
from PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi      import *
from PhysicsTools.PatAlgos.slimming.slimmedV0s_cff      import *
from PhysicsTools.PatAlgos.slimming.metFilterPaths_cff   import *
from PhysicsTools.PatAlgos.slimming.MicroEventContent_cff import *
from RecoEgamma.EgammaPhotonProducers.reducedEgamma_cfi  import *
from RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi import bunchSpacingProducer
from HeavyFlavorAnalysis.Onia2MuMu.OniaPhotonConversionProducer_cfi import PhotonCandidates as oniaPhotonCandidates

slimmingTask = cms.Task(
    packedPFCandidatesTask,
    lostTracks,
    isolatedTracks,
    offlineSlimmedPrimaryVertices,
    primaryVertexAssociation,
    genParticlesTask,
    selectedPatTrigger,
    slimmedPatTrigger,
    slimmedCaloJets,
    slimmedJets,
    slimmedJetsAK8,
    slimmedGenJets,
    slimmedGenJetsAK8,
    slimmedElectrons,
    slimmedLowPtElectrons,
    lowPtGsfLinks,
    slimmedMuons,
    slimmedPhotons,
    slimmedOOTPhotons,
    slimmedTaus,
    slimmedSecondaryVertices,
    slimmedKshortVertices,
    slimmedLambdaVertices,
    slimmedMETs,
    metFilterPathsTask,
    reducedEgamma,
    bunchSpacingProducer,
    oniaPhotonCandidates
)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toReplaceWith(slimmingTask, slimmingTask.copyAndExclude([slimmedOOTPhotons]))

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
_phase2_timing_slimmingTask = cms.Task(slimmingTask.copy(),
                                       offlineSlimmedPrimaryVertices4D)
phase2_timing.toReplaceWith(slimmingTask,_phase2_timing_slimmingTask)
