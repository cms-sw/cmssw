import FWCore.ParameterSet.Config as cms

from Geometry.EcalMapping.EcalMapping_cfi import *
eegeom = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalMappingRcd'),
    iovIsRunNotTime = cms.bool(False),
    firstValid = cms.vuint32(0)
)

ecalEBunpacker = cms.EDFilter("EcalRawToDigiDev",
    EcalFirstFED = cms.untracked.int32(12),
    srpUnpacking = cms.untracked.bool(False),
    headerUnpacking = cms.untracked.bool(False),
    DCCMapFile = cms.untracked.string('EventFilter/EcalRawToDigiDev/data/DCCMap.txt'),
    # Default is true
    eventPut = cms.untracked.bool(True),
    feUnpacking = cms.untracked.bool(True),
    #   untracked string InputLabel = "rawDataCollector"
    FEDs = cms.untracked.vint32(13),
    memUnpacking = cms.untracked.bool(False)
)


