import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the CSC geometry model.
#
# modelling flags (for completeness - internal defaults are already sane)
CSCGeometryESModule = cms.ESProducer("CSCGeometryESModule",
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(True),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.untracked.bool(False)
)


