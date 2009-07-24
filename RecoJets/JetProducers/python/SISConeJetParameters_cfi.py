import FWCore.ParameterSet.Config as cms

# Standard SISCone jets parameters
# $Id: SISConeJetParameters_cfi.py,v 1.2 2008/04/21 03:29:18 rpw Exp $
SISConeJetParameters = cms.PSet(
    protojetPtMin = cms.double(0.0),
    
    coneOverlapThreshold = cms.double(0.75),
    caching = cms.bool(False),
    maxPasses = cms.int32(0),
    splitMergeScale = cms.string('pttilde')
)

