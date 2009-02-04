import FWCore.ParameterSet.Config as cms

# Standard SISCone jets parameters
# $Id: SISConeJetParameters_cfi.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
SISConeJetParameters = cms.PSet(
    protojetPtMin = cms.double(0.0),
    
    coneOverlapThreshold = cms.double(0.75),
    caching = cms.bool(False),
    maxPasses = cms.int32(0),
    splitMergeScale = cms.string('pttilde')
)

