import FWCore.ParameterSet.Config as cms

# $Id: RecoJetAssociations_cff.py,v 1.7 2012/07/11 14:11:59 srappocc Exp $
#
# Associate reconstructed jets with other objects
# keep IC5 association for backward compatibility for external use
#from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
# standard associations
from RecoJets.JetAssociationProducers.ak5JTA_cff import *
recoJetAssociations = cms.Sequence(ak5JTA)

recoJetAssociationsExplicit = cms.Sequence(ak5JTAExplicit)
