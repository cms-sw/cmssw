import FWCore.ParameterSet.Config as cms

# $Id: ic5JetTracksAssociatorAtVertex_cfi.py,v 1.3 2010/02/17 17:47:51 wmtan Exp $
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ic5JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("iterativeCone5CaloJets")
)


