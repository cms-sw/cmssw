import FWCore.ParameterSet.Config as cms 

#this module re-makes the rec-hits using the global (weights+ratio) reco for hits saved in ecal selected digis 
from RecoEgamma.EgammaTools.ecalGlobalRecHitFromSelectedDigis_cff import *
#this module makes a new collection of barrel rechits where gain switched multifit crystals are swapped
#with weights reco hits if availible
from RecoEcal.EgammaClusterProducers.ecalMultiAndGSGlobalRecHitEB_cfi import *
#this sequence re-runs PF clustering with "GSFixed" suffext
from RecoEcal.EgammaClusterProducers.gsFixedSuperClustering_cff import *
#this module remakes the refined EGamma superclusters although it has to approximate them as there is not
#enough info in AOD to properly remake them
from RecoEcal.EgammaClusterProducers.particleFlowEGammaGSFixed_cfi import particleFlowEGammaGSFixed
#this makes makes a new colleciton of gsfelectron cores, modifying only those that have a gs eb crystal
from RecoEgamma.EgammaElectronProducers.gedGsfElectronCoresGSFixed_cfi import gedGsfElectronCoresGSFixed
#turns the cores into gsf electrons, again only modifying those which have a gs eb crystal
from RecoEgamma.EgammaElectronProducers.gedGsfElectronsGSFixed_cfi import gedGsfElectronsGSFixed
#this makes makes a new colleciton of photon cores, modifying only those that have a gs eb crystal
from RecoEgamma.EgammaPhotonProducers.gedPhotonCoreGSFixed_cfi import gedPhotonCoreGSFixed
#turns the cores into photons, again only modifying those which have a gs eb crystal
from RecoEgamma.EgammaPhotonProducers.gedPhotonsGSFixed_cfi import gedPhotonsGSFixed

egammaGainSwitchLocalFixSequence = cms.Sequence(
    ecalGlobalLocalRecoFromSelectedDigis*
    ecalMultiAndGSGlobalRecHitEB
)

egammaGainSwitchFixSequence = cms.Sequence(
    egammaGainSwitchLocalFixSequence*
    gsFixedParticleFlowSuperClustering*
    particleFlowEGammaGSFixed*
    gedGsfElectronCoresGSFixed*
    gedGsfElectronsGSFixed*
    gedPhotonCoreGSFixed*
    gedPhotonsGSFixed
    )
