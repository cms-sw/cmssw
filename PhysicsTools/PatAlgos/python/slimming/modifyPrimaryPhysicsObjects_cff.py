import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.modifiedElectrons_cfi import *
from PhysicsTools.PatAlgos.slimming.modifiedPhotons_cfi import *
from PhysicsTools.PatAlgos.slimming.modifiedMuons_cfi import *
from PhysicsTools.PatAlgos.slimming.modifiedTaus_cfi import *
from PhysicsTools.PatAlgos.slimming.modifiedJets_cfi import *

modifyPrimaryPhysicsObjects = cms.Sequence( slimmedElectrons *
                                            slimmedPhotons   *
                                            slimmedMuons     *
                                            slimmedTaus      *
                                            slimmedJets        )
