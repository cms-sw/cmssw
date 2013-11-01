import FWCore.ParameterSet.Config as cms

# Cambridge-Aachen top jet producer parameters
# $Id
CMSBoostedTauSeedingParameters = cms.PSet(
    useCMSBoostedTauSeedingAlgorithm = cms.bool(True),
    subjetPtMin = cms.double(10.0), # minimum subjet pt
    muMin = cms.double(0.0),        # minimum mass drop
    muMax = cms.double(0.667),      # maximum mass drop
    yMin = cms.double(-1.e+6),      # minimum asymmetry
    yMax = cms.double(+1.e+6),      # maximum asymmetry
    dRMin = cms.double(0.0),        # minimum delta R between subjets
    dRMax = cms.double(0.8),        # maximum delta R between subjets
    maxDepth = cms.int32(4)         # maximum depth for descending into clustering sequence
)

