import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.electronEcalRecHitIsolationLcone_cfi import *
from RecoEgamma.EgammaIsolationAlgos.electronEcalRecHitIsolationScone_cfi import *
from RecoEgamma.EgammaIsolationAlgos.electronHcalTowerIsolationLcone_cfi import *
from RecoEgamma.EgammaIsolationAlgos.electronHcalTowerIsolationScone_cfi import *
from RecoEgamma.EgammaIsolationAlgos.electronTrackIsolationLcone_cfi import *
from RecoEgamma.EgammaIsolationAlgos.electronTrackIsolationScone_cfi import *

#Standard reco sequence with both electrons and photons
egammaIsolationSequence = cms.Sequence(
    electronEcalRecHitIsolationLcone + 
    electronEcalRecHitIsolationScone + 
    electronHcalTowerIsolationLcone + 
    electronHcalTowerIsolationScone + 
    electronTrackIsolationLcone + 
    electronTrackIsolationScone
)


