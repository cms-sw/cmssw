import FWCore.ParameterSet.Config as cms

#
#  Author: M. Gouzevitch base on N. Marinelli work
#
pfConversions = cms.EDProducer("PFConversionProducer", 
    conversionCollection = cms.InputTag("allConversions", ""),
    PrimaryVertexLabel = cms.InputTag("offlinePrimaryVertices")  
)


# foo bar baz
# phqRflVLSrNAD
