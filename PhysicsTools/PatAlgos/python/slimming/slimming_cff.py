import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.packedPFCandidates_cff import *
from PhysicsTools.PatAlgos.slimming.isolatedTracks_cfi import *
from PhysicsTools.PatAlgos.slimming.lostTracks_cfi import *
from PhysicsTools.PatAlgos.slimming.offlineSlimmedPrimaryVertices_cfi import *
from PhysicsTools.PatAlgos.slimming.offlineSlimmedPrimaryVertices4D_cfi import *
from PhysicsTools.PatAlgos.slimming.offlineSlimmedPrimaryVerticesWithBS_cfi import *
from CommonTools.RecoAlgos.primaryVertexAssociation_cfi import *
from CommonTools.ParticleFlow.pfEGammaToCandidateRemapper_cfi import *
from PhysicsTools.PatAlgos.slimming.genParticles_cff import *
from PhysicsTools.PatAlgos.slimming.genParticleAssociation_cff import *
from PhysicsTools.PatAlgos.slimming.selectedPatTrigger_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedPatTrigger_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedJets_cfi      import *
from PhysicsTools.PatAlgos.slimming.slimmedCaloJets_cfi  import *
from PhysicsTools.PatAlgos.slimming.slimmedGenJets_cfi   import *
from PhysicsTools.PatAlgos.slimming.slimmedElectrons_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedLowPtElectrons_cff import *
from PhysicsTools.PatAlgos.slimming.slimmedTrackExtras_cff import *
from PhysicsTools.PatAlgos.slimming.slimmedMuons_cfi     import *
from PhysicsTools.PatAlgos.slimming.slimmedDisplacedMuons_cfi     import *
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
from RecoLocalCalo.HcalRecProducers.HcalHitSelection_cfi import *

slimmingTask = cms.Task(
    packedPFCandidatesTask,
    lostTracks,
    isolatedTracks,
    offlineSlimmedPrimaryVertices,
    offlineSlimmedPrimaryVerticesWithBS,
    primaryVertexAssociation,
    primaryVertexWithBSAssociation,
    genParticlesTask,
    packedCandidateToGenAssociationTask,
    selectedPatTrigger,
    slimmedPatTrigger,
    slimmedCaloJets,
    slimmedJets,
    slimmedJetsAK8,
    slimmedGenJets,
    slimmedGenJetsAK8,
    slimmedElectrons,
    slimmedLowPtElectronsTask,
    slimmedMuonTrackExtras,
    slimmedMuons,
    slimmedDisplacedMuonTrackExtras,
    slimmedDisplacedMuons,
    slimmedPhotons,
    slimmedOOTPhotons,
    slimmedTaus,
    slimmedSecondaryVertices,
    slimmedKshortVertices,
    slimmedLambdaVertices,
    slimmedMETs,
    metFilterPathsTask,
    reducedEgamma,
    slimmedHcalRecHits,
    bunchSpacingProducer,
    oniaPhotonCandidates
)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith(slimmingTask, slimmingTask.copyAndExclude([slimmedOOTPhotons]))

from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
_mAOD = (run2_miniAOD_94XFall17 | run2_miniAOD_80XLegacy)
(pp_on_AA | _mAOD).toReplaceWith(slimmingTask,
                                 slimmingTask.copyAndExclude([slimmedLowPtElectronsTask]))

from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
from Configuration.Eras.Era_Run2_2016_HIPM_cff import Run2_2016_HIPM
(pp_on_AA | _mAOD | run2_miniAOD_UL | Run2_2016_HIPM).toReplaceWith(slimmingTask,
                                                   slimmingTask.copyAndExclude([slimmedDisplacedMuons, slimmedDisplacedMuonTrackExtras]))

from PhysicsTools.PatAlgos.slimming.hiPixelTracks_cfi import hiPixelTracks
from RecoHI.HiEvtPlaneAlgos.HiEvtPlane_cfi import hiEvtPlane
from RecoHI.HiEvtPlaneAlgos.hiEvtPlaneFlat_cfi import hiEvtPlaneFlat
pp_on_AA.toReplaceWith(slimmingTask, cms.Task(slimmingTask.copy(), hiPixelTracks, hiEvtPlane, hiEvtPlaneFlat))

from PhysicsTools.PatAlgos.packedCandidateMuonID_cfi import packedCandidateMuonID
from PhysicsTools.PatAlgos.packedPFCandidateTrackChi2_cfi import packedPFCandidateTrackChi2
from RecoHI.HiCentralityAlgos.CentralityBin_cfi import centralityBin
from RecoHI.HiCentralityAlgos.hiHFfilters_cfi import hiHFfilters
lostTrackChi2 = packedPFCandidateTrackChi2.clone(candidates = "lostTracks", doLostTracks = True)

pp_on_AA.toReplaceWith(
    slimmingTask,
    cms.Task(slimmingTask.copy(), packedCandidateMuonID, packedPFCandidateTrackChi2, lostTrackChi2, centralityBin, hiHFfilters))
from Configuration.ProcessModifiers.run2_miniAOD_pp_on_AA_103X_cff import run2_miniAOD_pp_on_AA_103X
run2_miniAOD_pp_on_AA_103X.toReplaceWith(slimmingTask,cms.Task(primaryVertexAssociationCleaned,slimmingTask.copy()))
run2_miniAOD_pp_on_AA_103X.toReplaceWith(slimmingTask,cms.Task(primaryVertexWithBSAssociationCleaned,slimmingTask.copy()))
run2_miniAOD_pp_on_AA_103X.toReplaceWith(slimmingTask,cms.Task(pfEGammaToCandidateRemapperCleaned,slimmingTask.copy()))

from RecoHI.HiTracking.miniAODVertexRecovery_cff import offlinePrimaryVerticesRecovery, offlineSlimmedPrimaryVerticesRecovery
pp_on_AA.toReplaceWith(
    slimmingTask,
    cms.Task(slimmingTask.copy(), offlinePrimaryVerticesRecovery, offlineSlimmedPrimaryVerticesRecovery))

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(slimmingTask, slimmingTask.copyAndExclude([slimmedDisplacedMuons, slimmedDisplacedMuonTrackExtras]))

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
_phase2_timing_slimmingTask = cms.Task(slimmingTask.copy(),
                                       offlineSlimmedPrimaryVertices4D)
phase2_timing.toReplaceWith(slimmingTask,_phase2_timing_slimmingTask)

from PhysicsTools.PatAlgos.slimming.patPhotonDRNCorrector_cfi import patPhotonsDRN
from Configuration.ProcessModifiers.photonDRN_cff import _photonDRN
_photonDRN.toReplaceWith(slimmingTask, cms.Task(slimmingTask.copy(), patPhotonsDRN))
