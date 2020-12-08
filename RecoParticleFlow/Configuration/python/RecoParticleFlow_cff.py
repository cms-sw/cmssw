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

from RecoParticleFlow.PFTracking.hgcalTrackCollection_cfi import *
from RecoParticleFlow.PFProducer.simPFProducer_cff import *
from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import *
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *
particleFlowTmpBarrel = particleFlowTmp.clone()
_phase2_hgcal_particleFlowTmp = cms.EDProducer(
    "PFCandidateListMerger",
    src = cms.VInputTag("particleFlowTmpBarrel",
                        "simPFProducer")
    
)

_phase2_hgcal_simPFTask = cms.Task( pfTrack ,
                                    hgcalTrackCollection , 
                                    tpClusterProducer ,
                                    quickTrackAssociatorByHits ,
                                    simPFProducer )
_phase2_hgcal_simPFSequence = cms.Sequence(_phase2_hgcal_simPFTask) 
_phase2_hgcal_particleFlowRecoTask = cms.Task( _phase2_hgcal_simPFTask , particleFlowRecoTask.copy() )
_phase2_hgcal_particleFlowRecoTask.replace( particleFlowTmpTask, cms.Task( particleFlowTmpBarrel, particleFlowTmp ) )

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith( particleFlowTmp, _phase2_hgcal_particleFlowTmp )
phase2_hgcal.toReplaceWith( particleFlowRecoTask, _phase2_hgcal_particleFlowRecoTask )

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

for e in [pp_on_XeXe_2017, pp_on_AA]:
    e.toModify(particleFlowDisplacedVertexCandidate,
               tracksSelectorParameters = dict(pt_min = 999999.0,
                                               nChi2_max = 0.0,
                                               pt_min_prim = 999999.0,
                                               dxy = 999999.0)
               )

    e.toModify(particleFlowBlock, useNuclear = cms.bool(False))

    e.toModify(pfNoPileUpIso, enable = cms.bool(False))
    e.toModify(pfPileUpIso, enable = cms.bool(False))
    e.toModify(pfNoPileUp, enable = cms.bool(False))
    e.toModify(pfPileUp, enable = cms.bool(False))
    


