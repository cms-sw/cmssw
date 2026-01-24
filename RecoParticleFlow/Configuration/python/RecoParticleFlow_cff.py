import FWCore.ParameterSet.Config as cms


from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *
#from RecoParticleFlow.PFTracking.particleFlowTrackWithDisplacedVertex_cff import *

from RecoParticleFlow.PFProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFProducer.particleFlowBlock_cff import *

from RecoParticleFlow.PFProducer.particleFlowEGamma_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cff import *
from RecoParticleFlow.PFProducer.pfElectronTranslator_cff import *
from RecoParticleFlow.PFProducer.pfPhotonTranslator_cff import *
#from RecoParticleFlow.PFProducer.pfGsfElectronCiCSelector_cff import *
from RecoParticleFlow.PFProducer.pfGsfElectronMVASelector_cff import *

from RecoParticleFlow.PFProducer.pfLinker_cff import *

from CommonTools.ParticleFlow.pfParticleSelection_cff import *

from RecoEgamma.EgammaIsolationAlgos.particleBasedIsoProducer_cff import *
from RecoParticleFlow.PFProducer.chargedHadronPFTrackIsolation_cfi import *

from RecoJets.JetProducers.fixedGridRhoProducerFastjet_cfi import *
fixedGridRhoFastjetAllTmp = fixedGridRhoFastjetAll.clone(pfCandidatesTag = "particleFlowTmp")

particleFlowTmpTask = cms.Task(particleFlowTmp)
particleFlowTmpSeq = cms.Sequence(particleFlowTmpTask)

particleFlowRecoTask = cms.Task( particleFlowTrackWithDisplacedVertexTask,
#                                pfGsfElectronCiCSelectionSequence,
                                 pfGsfElectronMVASelectionTask,
                                 particleFlowBlock,
                                 particleFlowEGammaFullTask,
                                 particleFlowTmpTask,
                                 fixedGridRhoFastjetAllTmp,
                                 particleFlowTmpPtrs,
                                 particleFlowEGammaFinalTask,
                                 pfParticleSelectionTask )
particleFlowReco = cms.Sequence(particleFlowRecoTask)

particleFlowLinksTask = cms.Task( particleFlow, particleFlowPtrs, chargedHadronPFTrackIsolation, particleBasedIsolationTask)
particleFlowLinks = cms.Sequence(particleFlowLinksTask)

#
# for phase 2
particleFlowTmpBarrel = particleFlowTmp.clone()
_phase2_hgcal_particleFlowTmp = cms.EDProducer(
    "PFCandidateListMerger",
    src = cms.VInputTag("particleFlowTmpBarrel",
                        "pfTICL")

)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith( particleFlowTmp, _phase2_hgcal_particleFlowTmp )
phase2_hgcal.toModify(
    particleFlowTmpBarrel,
    vetoEndcap = True
    # If true, PF(Muon)Algo will ignore muon candidates incorporated via pfTICL
    # in addMissingMuons. This will prevent potential double-counting.
)
# We copy the standard task to create the Phase-2 specific task
_phase2_hgcal_particleFlowRecoTask = particleFlowRecoTask.copy()

# We replace the standard 'particleFlowTmpTask' (which runs standard PF)
# with a task that runs:
# 1. particleFlowTmpBarrel (Standard PF restricted to Barrel)
# 2. particleFlowTmp (The Merger of Barrel + TICL)
_phase2_hgcal_particleFlowRecoTask.replace(
    particleFlowTmpTask, 
    cms.Task(particleFlowTmpBarrel, particleFlowTmp)
)

# Switch to this new task in the phase2_hgcal era
phase2_hgcal.toReplaceWith( particleFlowRecoTask, _phase2_hgcal_particleFlowRecoTask )

#
# for heavy ion
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

for e in [pp_on_XeXe_2017, pp_on_AA]:
    e.toModify(particleFlowDisplacedVertexCandidate,
               tracksSelectorParameters = dict(pt_min = 999999.0,
                                               nChi2_max = 0.0,
                                               pt_min_prim = 999999.0,
                                               dxy = 999999.0)
               )
    e.toModify(pfNoPileUpIso, enable = False)
    e.toModify(pfPileUpIso, enable = False)
    e.toModify(pfNoPileUp, enable = False)
    e.toModify(pfPileUp, enable = False)


# for MLPF, replace standard PFAlgo with the ONNX-based MLPF producer 
from Configuration.ProcessModifiers.mlpf_cff import mlpf
from RecoParticleFlow.PFProducer.mlpfProducer_cfi import mlpfProducer
mlpf.toReplaceWith(particleFlowTmp, mlpfProducer)

#
# switch from pfTICL to simPF
def _findIndicesByModule(process,name):
    ret = []
    if hasattr(process,'particleFlowBlock'):
        for i, pset in enumerate(process.particleFlowBlock.elementImporters):
            if pset.importerName.value() == name:
                ret.append(i)
    return ret

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
def replaceTICLwithSimPF(process):
    if hasattr(process,'particleFlowTmp'):
      process.particleFlowTmp.src = ['particleFlowTmpBarrel', 'simPFProducer']

    if hasattr(process,'particleFlowTmpBarrel'):
      process.particleFlowTmpBarrel.vetoEndcap = False

    _insertTrackImportersWithVeto = {}
    _trackImporters = ['GeneralTracksImporter','ConvBremTrackImporter',
                   'ConversionTrackImporter','NuclearInteractionTrackImporter']
    for importer in _trackImporters:
        for idx in _findIndicesByModule(process,importer):
            _insertTrackImportersWithVeto[idx] = dict(
                vetoMode = cms.uint32(0), # HGCal-region PFTrack list for simPF
                vetoSrc = cms.InputTag('hgcalTrackCollection:TracksInHGCal')
            )
    phase2_hgcal.toModify(
      process.particleFlowBlock,
      elementImporters = _insertTrackImportersWithVeto
    )

    return process
