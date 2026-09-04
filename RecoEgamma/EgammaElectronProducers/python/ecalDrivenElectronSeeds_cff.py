import FWCore.ParameterSet.Config as cms

#
# module to produce pixel seeds for electrons from super clusters
# Author:  Ursula Berthon, Claude Charlot
#

from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeeds_cfi import ecalDrivenElectronSeeds
from RecoTracker.IterativeTracking.ElectronSeeds_cff import newCombinedSeeds

ecalDrivenElectronSeeds.initialSeedsVector = newCombinedSeeds.seedCollections

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.Eras.Modifier_run3_common_cff import run3_common
(pp_on_AA & ~run3_common).toModify(ecalDrivenElectronSeeds, SCEtCut = 15.0)
pp_on_AA.toModify(ecalDrivenElectronSeeds, maxHOverEBarrel = 0.5, maxHOverEEndcaps = 0.5)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(
    ecalDrivenElectronSeeds,
    endcapSuperClusters = 'particleFlowSuperClusterHGCal',
    allowHGCal = True,
)
phase2_hgcal.toModify(
    ecalDrivenElectronSeeds.HGCalConfig,
    hgcalHitMap = cms.InputTag("recHitMapProducer", "hgcalRecHitMap"),
)

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(ecalDrivenElectronSeeds,
                           LowPtThreshold =1.0,
                           applyHOverECut = False) 

