import FWCore.ParameterSet.Config as cms


from RecoJets.JetAssociationProducers.jvcParameters_cfi import *

ic5JetVertexCompatibility = cms.EDProducer("JetSignalVertexCompatibility",
	jvcParameters,
	jetTracksAssoc = cms.InputTag("ic5JetTracksAssociatorAtVertex"),
)
# foo bar baz
# JhD1neraflk6k
# mkJFJCfwr3rQu
