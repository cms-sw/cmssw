import FWCore.ParameterSet.Config as cms

#
# sequence to make photons from clusters in ECAL
#
# photon producer
from RecoEgamma.EgammaPhotonProducers.photonCore_cfi import *
from RecoEgamma.EgammaPhotonProducers.photons_cfi import *

photonTask = cms.Task(photonCore,photons)
photonSequence = cms.Sequence(photonTask)

_photonTaskFromMultiCl = photonTask.copy()
_photonTaskFromMultiCl.add(photonCoreFromMultiCl,photonsFromMultiCl)
_photonTaskWithIsland = photonTask.copy()
_photonTaskWithIsland.add(islandPhotonCore,islandPhotons)


from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith(
 photonTask, _photonTaskFromMultiCl
)

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
for e in [pA_2016, peripheralPbPb, pp_on_AA_2018, pp_on_XeXe_2017, ppRef_2017, pp_on_PbPb_run3]:
    e.toReplaceWith(photonTask, _photonTaskWithIsland)
