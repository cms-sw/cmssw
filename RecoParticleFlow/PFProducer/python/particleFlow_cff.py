import FWCore.ParameterSet.Config as cms

#Geometry
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cfi import *
from RecoParticleFlow.PFProducer.particleFlowTmpPtrs_cfi import *

particleFlowTmp = particleFlow.clone()

## temporary for 12_1; EtaExtendedEles do not enter PF because ID/regression of EEEs are not ready yet
## In 12_2, we expect to have EEE's ID/regression, then this switch can flip to True
particleFlowTmp.PFEGammaFiltersParameters.allowEEEinPF = cms.bool(False)

from Configuration.Eras.Modifier_pf_badHcalMitigationOff_cff import pf_badHcalMitigationOff
pf_badHcalMitigationOff.toModify(particleFlowTmp.PFEGammaFiltersParameters,
                                 electron_protectionsForBadHcal = dict(enableProtections = False),
                                 photon_protectionsForBadHcal   = dict(enableProtections = False))

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(particleFlowTmp.PFEGammaFiltersParameters,photon_MinEt = 1.)
