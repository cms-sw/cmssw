import FWCore.ParameterSet.Config as cms

#------------------------------------------------------------
# This cfi is a special version for use only when comparing
# with old orca simulations in which the real csc wire
# geometry was not modelled. this is flagged by the setting
# useRealWireGeometry=false
#
# This was the case for 'most' simulation performed with orca.
#
# You'll also need an appropriate cscSpecs.xml file:
# Geometry/CSCGeometryBuilder/data/cscSpecsOldOrca.xml
# in place of the default version.
#------------------------------------------------------------
#
# This cfi should be included to build the CSC geometry model.
#
CSCGeometryESModule = cms.ESProducer("CSCGeometryESModule",
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(False),
    useRadialStrips = cms.bool(True),
    useGangedStripsInME1a = cms.bool(True),
    applyAlignment = cms.untracked.bool(False)
)


