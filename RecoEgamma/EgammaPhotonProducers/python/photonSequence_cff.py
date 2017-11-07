import FWCore.ParameterSet.Config as cms

#
# sequence to make photons from clusters in ECAL
#
# photon producer
from RecoEgamma.EgammaPhotonProducers.photonCore_cfi import *
from RecoEgamma.EgammaPhotonProducers.photons_cfi import *

photonSequence = cms.Sequence( photonCore + photons )
_photonSequenceFromMultiCl = photonSequence.copy()
_photonSequenceFromMultiCl += ( photonCoreFromMultiCl + photonsFromMultiCl)
_photonSequenceWithIsland = photonSequence.copy()
_photonSequenceWithIsland += ( islandPhotonCore + islandPhotons )


from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith(
 photonSequence, _photonSequenceFromMultiCl
)

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
for e in [pA_2016, peripheralPbPb, pp_on_XeXe_2017, ppRef_2017]:
    e.toReplaceWith(photonSequence, _photonSequenceWithIsland)
