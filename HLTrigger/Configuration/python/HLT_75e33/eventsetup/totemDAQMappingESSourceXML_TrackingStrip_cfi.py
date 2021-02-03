import FWCore.ParameterSet.Config as cms

totemDAQMappingESSourceXML_TrackingStrip = cms.ESSource("TotemDAQMappingESSourceXML",
    configuration = cms.VPSet(
        cms.PSet(
            mappingFileNames = cms.vstring('CondFormats/PPSObjects/xml/mapping_tracking_strip_2016_to_fill_5288.xml'),
            maskFileNames = cms.vstring(),
            validityRange = cms.EventRange(1, 0, 1, 280385, 0, 0)
        ),
        cms.PSet(
            mappingFileNames = cms.vstring(),
            maskFileNames = cms.vstring(),
            validityRange = cms.EventRange(280386, 0, 1, 281600, 0, 0)
        ),
        cms.PSet(
            mappingFileNames = cms.vstring('CondFormats/PPSObjects/xml/mapping_tracking_strip_2016_from_fill_5330.xml'),
            maskFileNames = cms.vstring(),
            validityRange = cms.EventRange(281601, 0, 1, 290872, 0, 0)
        ),
        cms.PSet(
            mappingFileNames = cms.vstring('CondFormats/PPSObjects/xml/mapping_tracking_strip_2017.xml'),
            maskFileNames = cms.vstring(),
            validityRange = cms.EventRange(290873, 0, 1, 311625, 0, 0)
        ),
        cms.PSet(
            mappingFileNames = cms.vstring('CondFormats/PPSObjects/xml/mapping_tracking_strip_2018.xml'),
            maskFileNames = cms.vstring(),
            validityRange = cms.EventRange(311626, 0, 1, 999999999, 0, 0)
        )
    ),
    subSystem = cms.untracked.string('TrackingStrip'),
    verbosity = cms.untracked.uint32(0)
)
