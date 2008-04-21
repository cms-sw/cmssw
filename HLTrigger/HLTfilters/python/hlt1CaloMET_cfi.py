# The following comments couldn't be translated into the new config version:

# negative = no cut
import FWCore.ParameterSet.Config as cms

hlt1CaloMET = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("met"),
    MinPt = cms.double(100.0),
    MinN = cms.int32(1)
)


