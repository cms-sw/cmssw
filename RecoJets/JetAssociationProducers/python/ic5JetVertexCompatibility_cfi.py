import FWCore.ParameterSet.Config as cms

# $Id: ic5JetVertexCompatibility_cfi.py,v 1.1 2009/03/26 10:35:00 saout Exp $

from RecoJets.JetAssociationProducers.jvcParameters_cfi import *

ic5JetVertexCompatibility = cms.EDProducer("JetSignalVertexCompatibility",
	jvcParameters,
	jetTracksAssoc = cms.InputTag("ic5JetTracksAssociatorAtVertex"),
)
