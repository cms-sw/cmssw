import FWCore.ParameterSet.Config as cms

#Geometry
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cfi import *
from RecoParticleFlow.PFProducer.particleFlowTmpPtrs_cfi import *

particleFlowTmp = particleFlow.clone()

# Thresholds for e/gamma PFID DNN 
# Thresholds for electron: Sig_isolated+Sig_nonIsolated
particleFlowTmp.PFEGammaFiltersParameters.electronDnnThresholds = cms.PSet(
            electronDnnHighPtBarrelThr = cms.double(0.068),
            electronDnnHighPtEndcapThr = cms.double(0.056),
            electronDnnLowPtThr = cms.double(0.075), 
            electronDnnExtEta1Thr = cms.double(0.113604), 
            electronDnnExtEta2Thr = cms.double(0.118044) 
        )
# Thresholds for electron: Bkg_nonIsolated
particleFlowTmp.PFEGammaFiltersParameters.electronDnnBkgThresholds = cms.PSet(
            electronDnnBkgHighPtBarrelThr = cms.double(0.8),
            electronDnnBkgHighPtEndcapThr = cms.double(0.75),
            electronDnnBkgLowPtThr = cms.double(0.75), 
            electronDnnBkgExtEta1Thr = cms.double(0.7), 
            electronDnnBkgExtEta2Thr = cms.double(0.85) 
        )
# Thresholds for photons
particleFlowTmp.PFEGammaFiltersParameters.photonDnnThresholds = cms.PSet(
            photonDnnBarrelThr = cms.double(0.22),
            photonDnnEndcapThr = cms.double(0.35)
)

from Configuration.Eras.Modifier_pf_badHcalMitigationOff_cff import pf_badHcalMitigationOff
pf_badHcalMitigationOff.toModify(particleFlowTmp.PFEGammaFiltersParameters,
                                 electron_protectionsForBadHcal = dict(enableProtections = False),
                                 photon_protectionsForBadHcal   = dict(enableProtections = False))

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(particleFlowTmp.PFEGammaFiltersParameters,photon_MinEt = 1.)

# Activate Egamma PFID with DNN for Run3
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(particleFlowTmp.PFEGammaFiltersParameters,
    useElePFidDnn = True,  
    usePhotonPFidDnn = True
)
