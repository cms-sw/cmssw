import FWCore.ParameterSet.Config as cms 

#this module re-makes the rec-hits using the weights reco for hits saved in ecal selected digis 
from RecoEgamma.EgammaTools.ecalWeightRecHitFromSelectedDigis_cff import *
#this module makes a new collection of barrel rechits where gain switched multifit crystals are swapped
#with weights reco hits if availible
from RecoEcal.EgammaClusterProducers.ecalMultiAndGSWeightRecHitEB_cfi import *
#this sequence re-runs PF clustering with "GSFixed" suffext
from RecoEcal.EgammaClusterProducers.gsFixedSuperClustering_cff import *
#this module remakes the refined EGamma superclusters although it has to approximate them as there is not
#enough info in AOD to properly remake them
from RecoEcal.EgammaClusterProducers.gsFixedRefinedBarrelSuperClusters_cfi import *
#this makes a make of old superclusters with the gs issue to new superclusters without the gs issue
from RecoEcal.EgammaClusterProducers.gsBrokenToGSFixedSuperClustersMap_cfi import *
#this makes makes a new colleciton of gsfelectron cores, modifying only those that have a gs eb crystal
from RecoEgamma.EgammaElectronProducers.gsFixedGsfElectronCores_cfi import *
#turns the cores into gsf electrons, again only modifying those which have a gs eb crystal
from RecoEgamma.EgammaElectronProducers.gsFixedGsfElectrons_cfi import *
#this makes makes a new colleciton of ged photon cores, modifying only those that have a gs eb crystal
from RecoEgamma.EgammaPhotonProducers.gsFixedGedPhotonCores_cfi import *
#turns the cores into ged photons, again only modifying those which have a gs eb crystal
from RecoEgamma.EgammaPhotonProducers.gsFixedGedPhotons_cfi import *

egammaGainSwitchFixSequence = cms.Sequence(
#    bunchSpacingProducer*
    ecalWeightLocalRecoFromSelectedDigis*
    ecalMultiAndGSWeightRecHitEB*
    gsFixedParticleFlowSuperClustering*
    gsFixedRefinedBarrelSuperClusters*
    gsBrokenToGSFixedSuperClustersMap*
    gsFixedGsfElectronCores*
    gsFixedGsfElectrons*
    gsFixedGedPhotonCores*
    gsFixedGedPhotons)

from RecoEgamma.EgammaElectronProducers.gsSimpleFixedGsfElectrons_cfi import gsSimpleFixedGsfElectrons
from RecoEgamma.EgammaElectronProducers.gsSimpleFixedPhotons_cfi import gsSimpleFixedPhotons

egammaGainSwitchSimpleFixSequence = cms.Sequence(
#    bunchSpacingProducer*
    ecalWeightLocalRecoFromSelectedDigis*
    ecalMultiAndGSWeightRecHitEB*
    gsSimpleFixedGsfElectrons*
    gsSimpleFixedPhotons)
