import FWCore.ParameterSet.Config as cms

# $Id: ic5JetTracksAssociatorAtVertex_cfi.py,v 1.2 2008/04/21 03:27:42 rpw Exp $
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ic5JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("iterativeCone5CaloJets")
)


