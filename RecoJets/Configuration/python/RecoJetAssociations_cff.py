import FWCore.ParameterSet.Config as cms

#
# Associate reconstructed jets with other objects
# keep IC5 association for backward compatibility for external use
from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import *
#from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
# standard associations
from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import *
from RecoJets.JetAssociationProducers.sisCone5JTA_cff import *
from RecoJets.JetAssociationProducers.kt4JTA_cff import *
from RecoJets.JetAssociationProducers.ak4JTA_cff import *
from RecoJets.JetAssociationProducers.ak7JTA_cff import *
recoJetAssociationsTask = cms.Task(ak4JTATask)
recoJetAssociations = cms.Sequence(recoJetAssociationsTask)
recoJetAssociationsExplicitTask = cms.Task(ak4JTAExplicitTask)
recoJetAssociationsExplicit = cms.Sequence(recoJetAssociationsExplicitTask)
