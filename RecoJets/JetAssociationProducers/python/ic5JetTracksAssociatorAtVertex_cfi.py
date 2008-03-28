import FWCore.ParameterSet.Config as cms

# $Id: ic5JetTracksAssociatorAtVertex.cfi,v 1.1 2007/09/20 22:32:40 fedor Exp $
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ic5JetTracksAssociatorAtVertex = cms.EDFilter("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("iterativeCone5CaloJets")
)


