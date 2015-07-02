import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.services.SiStripConfObjectGeneratorService_cfi import *

SiStripConfObjectGenerator.Parameters = cms.VPSet(
    cms.PSet(
        ParameterName = cms.string("defaultPartitionNames"),
        ParameterType = cms.string("vstring"),
        ParameterValue = cms.vstring("TI","TO","TP","TM"),
    ),
    cms.PSet(
        ParameterName = cms.string("defaultPhases"),
        ParameterType = cms.string("vint32"),
        ParameterValue = cms.vint32(63,63,63,63),
    ),
    cms.PSet(
        ParameterName = cms.string("useEC0"),
        ParameterType = cms.string("bool"),
        ParameterValue = cms.bool(False),
    ),
    cms.PSet(
        ParameterName = cms.string("badRun"),
        ParameterType = cms.string("bool"),
        ParameterValue = cms.bool(False),
    ),
    cms.PSet(
        ParameterName = cms.string("magicOffset"),
        ParameterType = cms.string("int"),
        ParameterValue = cms.int32(8),
    ),
)

siStripConfObjectAPVPhaseOffsetsFakeESSource = cms.ESSource("SiStripConfObjectFakeESSource",
                                                            appendToDataLabel = cms.string('apvphaseoffsets')
                                                            )



