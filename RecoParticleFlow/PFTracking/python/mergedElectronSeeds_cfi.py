import FWCore.ParameterSet.Config as cms


electronMergedSeeds =cms.EDProducer("ElectronSeedMerger",
     EcalBasedSeeds = cms.InputTag("ecalDrivenElectronSeeds"),
     TkBasedSeeds  = cms.InputTag("trackerDrivenElectronSeeds:SeedsForGsf")
    )

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(electronMergedSeeds, TkBasedSeeds = '')

electronMergedSeedsFromMultiCl = electronMergedSeeds.clone(
  EcalBasedSeeds = 'ecalDrivenElectronSeedsFromMultiCl'
)
