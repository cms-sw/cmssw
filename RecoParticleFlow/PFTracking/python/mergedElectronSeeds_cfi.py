import FWCore.ParameterSet.Config as cms
from copy import deepcopy


electronMergedSeeds =cms.EDProducer("ElectronSeedMerger",
     EcalBasedSeeds = cms.InputTag("ecalDrivenElectronSeeds"),
     TkBasedSeeds  = cms.InputTag("trackerDrivenElectronSeeds:SeedsForGsf")
    )

electronMergedSeedsFromMultiCl = deepcopy(electronMergedSeeds)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(
  electronMergedSeedsFromMultiCl,
  EcalBasedSeeds = 'ecalDrivenElectronSeedsFromMultiCl'
)
