import FWCore.ParameterSet.Config as cms

# Standard SISCone jets parameters
# $Id: SISConeJetParameters.cfi,v 1.2 2007/10/26 22:29:55 fedor Exp $
SISConeJetParameters = cms.PSet(
    protojetPtMin = cms.double(0.0),
    JetPtMin = cms.double(1.0),
    coneOverlapThreshold = cms.double(0.75),
    caching = cms.bool(False),
    maxPasses = cms.int32(0),
    splitMergeScale = cms.string('pttilde')
)

