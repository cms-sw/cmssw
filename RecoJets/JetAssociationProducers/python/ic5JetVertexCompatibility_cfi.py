import FWCore.ParameterSet.Config as cms

# $Id: ic5JetTracksAssociatorAtVertex_cfi.py,v 1.2 2008/04/21 03:27:42 rpw Exp $

from RecoJets.JetAssociationProducers.jvcParameters_cfi import *

ic5JetVertexCompatibility = cms.EDProducer("JetSignalVertexCompatibility",
	jvcParameters,
	jetTracksAssoc = cms.InputTag("ic5JetTracksAssociatorAtVertex"),
)
