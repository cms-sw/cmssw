import FWCore.ParameterSet.Config as cms

#Geometry
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cfi import *
from RecoParticleFlow.PFProducer.particleFlowTmpPtrs_cfi import *

particleFlowTmp = particleFlow.clone()

from Configuration.Eras.Modifier_pf_badHcalMitigation_cff import pf_badHcalMitigation
pf_badHcalMitigation.toModify(particleFlowTmp.PFEGammaFiltersParameters,
        electron_protectionsForBadHcal = dict(enableProtections = True),
        photon_protectionsForBadHcal   = dict(enableProtections = True))

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(particleFlowTmp.PFEGammaFiltersParameters,photon_MinEt = 1.)
