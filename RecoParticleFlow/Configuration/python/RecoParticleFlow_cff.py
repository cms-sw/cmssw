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

from RecoJets.JetProducers.fixedGridRhoProducerFastjet_cfi import *
fixedGridRhoFastjetAllTmp = fixedGridRhoFastjetAll.clone(pfCandidatesTag = cms.InputTag("particleFlowTmp"))

particleFlowTmpSeq = cms.Sequence(particleFlowTmp)

particleFlowReco = cms.Sequence( particleFlowTrackWithDisplacedVertex*
#                                pfGsfElectronCiCSelectionSequence*
                                 pfGsfElectronMVASelectionSequence*
                                 particleFlowBlock*
                                 particleFlowEGammaFull*
                                 particleFlowTmpSeq*
                                 fixedGridRhoFastjetAllTmp*
                                 particleFlowTmpPtrs*          
                                 particleFlowEGammaFinal*
                                 pfParticleSelectionSequence )

particleFlowLinks = cms.Sequence( particleFlow*particleFlowPtrs*particleBasedIsolationSequence)

from RecoParticleFlow.PFTracking.hgcalTrackCollection_cfi import *
from RecoParticleFlow.PFProducer.simPFProducer_cfi import *
from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import *
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *
particleFlowTmpBarrel = particleFlowTmp.clone()
_phase2_hgcal_particleFlowTmp = cms.EDProducer(
    "PFCandidateListMerger",
    src = cms.VInputTag("particleFlowTmpBarrel",
                        "simPFProducer")
    
)

_phase2_hgcal_simPFSequence = cms.Sequence( pfTrack +
                                            hgcalTrackCollection + 
                                            tpClusterProducer +
                                            quickTrackAssociatorByHits +
                                            simPFProducer )
_phase2_hgcal_particleFlowReco = cms.Sequence( _phase2_hgcal_simPFSequence * particleFlowReco.copy() )
_phase2_hgcal_particleFlowReco.replace( particleFlowTmpSeq, cms.Sequence( particleFlowTmpBarrel * particleFlowTmp ) )

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( quickTrackAssociatorByHits,
                            pixelSimLinkSrc = cms.InputTag("simSiPixelDigis","Pixel"),
                            stripSimLinkSrc = cms.InputTag("simSiPixelDigis","Tracker")
                            )

phase2_hgcal.toModify( tpClusterProducer,
                            pixelSimLinkSrc = cms.InputTag("simSiPixelDigis", "Pixel"),
                            phase2OTSimLinkSrc = cms.InputTag("simSiPixelDigis","Tracker")
                            )

phase2_hgcal.toReplaceWith( particleFlowTmp, _phase2_hgcal_particleFlowTmp )
phase2_hgcal.toReplaceWith( particleFlowReco, _phase2_hgcal_particleFlowReco )
