import FWCore.ParameterSet.Config as cms

#
#  run on collection of electrons to make a collection of AlCaReco electrons 
#  and store them in the output collection
#
# hybrid clustering in the barrel
from RecoEcal.EgammaClusterProducers.hybridClusteringSequence_cff import *
# island clustering for the endcaps
from RecoEcal.EgammaClusterProducers.islandClusteringSequence_cff import *
# sequence to make si-strip based electrons
from RecoEgamma.EgammaElectronProducers.electronSequence_cff import *
from Calibration.EcalAlCaRecoProducers.alCaIsolatedElectrons_cfi import *
alcaElectronSequence = cms.Sequence(hybridClusteringSequence*islandClusteringSequence*electronSequence*alCaIsolatedElectrons)

