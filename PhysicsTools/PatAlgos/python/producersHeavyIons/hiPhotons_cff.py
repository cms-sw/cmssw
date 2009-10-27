import FWCore.ParameterSet.Config as cms

# PAT sequence
from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cff import *
from PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi import *
from PhysicsTools.PatAlgos.recoLayer0.photonIsolation_cff import *

# HIEgammaIsolationSequence  (Must be decleared after PAT Sequence Yen-Jie)
from RecoHI.HiEgammaAlgos.HiEgammaIsolation_cff import *


# Use genParticles for the moment
photonMatch.matched = cms.InputTag("hiGenParticles")

allLayer1Photons.addPhotonID = cms.bool(True)
allLayer1Photons.addGenMatch = cms.bool(True)
allLayer1Photons.embedGenMatch = cms.bool(True)

# HI Photon Isolation
allLayer1Photons.userData.userFloats.src  = cms.VInputTag(
 cms.InputTag("isoCC1"),cms.InputTag("isoCC2"),cms.InputTag("isoCC3"),cms.InputTag("isoCC4"),cms.InputTag("isoCC5"),
 cms.InputTag("isoCR1"),cms.InputTag("isoCR2"),cms.InputTag("isoCR3"),cms.InputTag("isoCR4"),cms.InputTag("isoCR5"),
 cms.InputTag("isoT11"),cms.InputTag("isoT12"),cms.InputTag("isoT13"),cms.InputTag("isoT14"),  
 cms.InputTag("isoT21"),cms.InputTag("isoT22"),cms.InputTag("isoT23"),cms.InputTag("isoT24"),  
 cms.InputTag("isoT31"),cms.InputTag("isoT32"),cms.InputTag("isoT33"),cms.InputTag("isoT34"),  
 cms.InputTag("isoT41"),cms.InputTag("isoT42"),cms.InputTag("isoT43"),cms.InputTag("isoT44"),  
 cms.InputTag("isoDR11"),cms.InputTag("isoDR12"),cms.InputTag("isoDR13"),cms.InputTag("isoDR14"),  
 cms.InputTag("isoDR21"),cms.InputTag("isoDR22"),cms.InputTag("isoDR23"),cms.InputTag("isoDR24"),  
 cms.InputTag("isoDR31"),cms.InputTag("isoDR32"),cms.InputTag("isoDR33"),cms.InputTag("isoDR34"),  
 cms.InputTag("isoDR41"),cms.InputTag("isoDR42"),cms.InputTag("isoDR43"),cms.InputTag("isoDR44")
)

# Use Loose ID for the moment (pp algo)
allLayer1Photons.photonIDSource = cms.InputTag("PhotonIDProd","PhotonCutBasedIDLoose")

hiPatPhotonSequence = cms.Sequence(hiEgammaIsolationSequence*patPhotonIsolation*photonMatch*allLayer1Photons)

############################################
#
# HI PAT Photon - Selection
#

from PhysicsTools.PatAlgos.selectionLayer1.photonSelector_cfi import *
selectedLayer1Photons.cut = cms.string('pt > 0. & abs(eta) < 12.')


