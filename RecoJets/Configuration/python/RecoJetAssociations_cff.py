import FWCore.ParameterSet.Config as cms

#
# Associate reconstructed jets with other objects
#from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
# standard associations
from RecoJets.JetAssociationProducers.ak4JTA_cff import *
from RecoJets.JetAssociationProducers.kt4JTA_cff import *
from RecoJets.JetAssociationProducers.ak5JTA_cff import *
from RecoJets.JetAssociationProducers.ak8JTA_cff import *
recoJetAssociations = cms.Sequence(ak4JetTracksAssociatorAtVertex+ak4JTA)#+ak5JTA)
recoJetAssociationsExplicit = cms.Sequence(ak4JTAExplicit)
