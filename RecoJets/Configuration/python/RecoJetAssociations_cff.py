import FWCore.ParameterSet.Config as cms

#
# Associate reconstructed jets with other objects
# keep AK4 association for backward compatibility for external use
from RecoJets.JetAssociationProducers.ak4JetTracksAssociatorAtVertex_cfi import *
#from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
# standard associations
from RecoJets.JetAssociationProducers.ak4JTA_cff import *
from RecoJets.JetAssociationProducers.sisCone5JTA_cff import *
from RecoJets.JetAssociationProducers.kt4JTA_cff import *
from RecoJets.JetAssociationProducers.ak4JTA_cff import *
from RecoJets.JetAssociationProducers.ak8JTA_cff import *
recoJetAssociations = cms.Sequence(ak4JetTracksAssociatorAtVertex+ak4JTA+ak4JTA)
recoJetAssociationsExplicit = cms.Sequence(ak4JTAExplicit)
