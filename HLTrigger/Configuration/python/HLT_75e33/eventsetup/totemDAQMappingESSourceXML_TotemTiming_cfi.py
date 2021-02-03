import FWCore.ParameterSet.Config as cms

totemDAQMappingESSourceXML_TotemTiming = cms.ESSource("TotemDAQMappingESSourceXML",
    configuration = cms.VPSet(
        cms.PSet(
            mappingFileNames = cms.vstring(),
            maskFileNames = cms.vstring(),
            validityRange = cms.EventRange(1, 0, 1, 310000, 0, 0)
        ),
        cms.PSet(
            mappingFileNames = cms.vstring('CondFormats/PPSObjects/xml/mapping_totem_timing_2018.xml'),
            maskFileNames = cms.vstring(),
            validityRange = cms.EventRange(310001, 0, 1, 999999999, 0, 0)
        )
    ),
    subSystem = cms.untracked.string('TotemTiming'),
    verbosity = cms.untracked.uint32(10)
)
