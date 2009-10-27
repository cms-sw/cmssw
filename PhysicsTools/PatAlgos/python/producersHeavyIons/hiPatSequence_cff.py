import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersHeavyIons.hiPhotons_cff import *
from PhysicsTools.PatAlgos.producersHeavyIons.hiJets_cff import *
from PhysicsTools.PatAlgos.producersHeavyIons.hiMuons_cff import *
from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonProducer_cfi import *

hiPatProductionSequence = cms.Sequence(
    heavyIon *
    hiPatJetSequence *
    hiPatPhotonSequence *
    hiPatMuonSequence
    )

hiPatSelectionSequence = cms.Sequence(
    selectedLayer1Muons +
    selectedLayer1Photons +
    selectedLayer1Jets
    )

##########################################################
#
# Event Content - here until a dedicated file is created
#

hiPatEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *',
                                           'keep patPhotons_selected*_*_*',
                                           'keep patMuons_selected*_*_*',
                                           'keep patJets_selected*_*_*',
                                           'keep patHeavyIon_heavyIon_*_*'
                                           )
    )

hiPatExtra = cms.PSet( outputCommands = cms.untracked.vstring('keep recoGenParticles_hiGenParticles_*_*',
                                                              'keep recoGenJets_iterativeCone5HiGenJets_*_*', # until a better solution
                                                              'keep recoTracks_hiSelectedTracks_*_*'
                                                              ))

hiPatExtraEventContent = hiPatEventContent.clone()
hiPatExtraEventContent.outputCommands.extend(hiPatExtra.outputCommands)

