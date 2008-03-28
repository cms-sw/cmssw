import FWCore.ParameterSet.Config as cms

# $Id: RecoJetAssociations.cff,v 1.5 2008/02/29 00:42:40 fedor Exp $
#
# Associate reconstructed jets with other objects
# keep IC5 association for backward compatibility for external use
from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import *
# standard associations
from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import *
from RecoJets.JetAssociationProducers.sisCone5JTA_cff import *
from RecoJets.JetAssociationProducers.kt4JTA_cff import *
recoJetAssociations = cms.Sequence(ic5JetTracksAssociatorAtVertex*iterativeCone5JTA+sisCone5JTA+kt4JTA)

