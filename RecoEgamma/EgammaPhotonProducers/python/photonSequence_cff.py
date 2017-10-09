import FWCore.ParameterSet.Config as cms

#
# sequence to make photons from clusters in ECAL
#
# photon producer
from RecoEgamma.EgammaPhotonProducers.photonCore_cfi import *
from RecoEgamma.EgammaPhotonProducers.photons_cfi import *

photonSequence = cms.Sequence( photonCore + photons )
_photonSequenceFromMultiCl = photonSequence.copy()
_photonSequenceFromMultiCl += cms.Sequence ( photonCoreFromMultiCl + photonsFromMultiCl)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith(
 photonSequence, _photonSequenceFromMultiCl
)
