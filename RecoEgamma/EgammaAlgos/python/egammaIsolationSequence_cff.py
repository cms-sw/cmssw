import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaAlgos.electronEcalRecHitIsolationLcone_cfi import *
from RecoEgamma.EgammaAlgos.electronEcalRecHitIsolationScone_cfi import *
from RecoEgamma.EgammaAlgos.electronHcalTowerIsolationLcone_cfi import *
from RecoEgamma.EgammaAlgos.electronHcalTowerIsolationScone_cfi import *
from RecoEgamma.EgammaAlgos.electronTrackIsolationLcone_cfi import *
from RecoEgamma.EgammaAlgos.electronTrackIsolationScone_cfi import *

#Standard reco sequence with both electrons and photons
egammaIsolationSequence = cms.Sequence(
    electronEcalRecHitIsolationLcone + 
    electronEcalRecHitIsolationScone + 
    electronHcalTowerIsolationLcone +
    electronHcalDepth1TowerIsolationLcone +
    electronHcalDepth2TowerIsolationLcone +
    electronHcalTowerIsolationScone +
    electronHcalDepth1TowerIsolationScone +
    electronHcalDepth2TowerIsolationScone +
    electronTrackIsolationLcone + 
    electronTrackIsolationScone
)


