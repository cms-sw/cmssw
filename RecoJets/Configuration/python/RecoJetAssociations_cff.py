import FWCore.ParameterSet.Config as cms

# $Id: RecoJetAssociations_cff.py,v 1.3 2009/07/31 03:27:31 srappocc Exp $
#
# Associate reconstructed jets with other objects
# keep IC5 association for backward compatibility for external use
from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import *
# standard associations
from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import *
from RecoJets.JetAssociationProducers.sisCone5JTA_cff import *
from RecoJets.JetAssociationProducers.kt4JTA_cff import *
from RecoJets.JetAssociationProducers.ak5JTA_cff import *
from RecoJets.JetAssociationProducers.ak7JTA_cff import *
recoJetAssociations = cms.Sequence(ic5JetTracksAssociatorAtVertex*iterativeCone5JTA+sisCone5JTA+kt4JTA+ak5JTA+ak7JTA)

