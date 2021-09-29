import FWCore.ParameterSet.Config as cms

totemDAQMappingESSourceXML_TimingDiamond = cms.ESSource("TotemDAQMappingESSourceXML",
    configuration = cms.VPSet(
        cms.PSet(
            mappingFileNames = cms.vstring(),
            maskFileNames = cms.vstring(),
            validityRange = cms.EventRange(1, 0, 1, 283819, 0, 0)
        ),
        cms.PSet(
            mappingFileNames = cms.vstring('CondFormats/PPSObjects/xml/mapping_timing_diamond.xml'),
            maskFileNames = cms.vstring(),
            validityRange = cms.EventRange(283820, 0, 1, 292520, 0, 0)
        ),
        cms.PSet(
            mappingFileNames = cms.vstring('CondFormats/PPSObjects/xml/mapping_timing_diamond_2017.xml'),
            maskFileNames = cms.vstring(),
            validityRange = cms.EventRange(292521, 0, 1, 310000, 0, 0)
        ),
        cms.PSet(
            mappingFileNames = cms.vstring('CondFormats/PPSObjects/xml/mapping_timing_diamond_2018.xml'),
            maskFileNames = cms.vstring(),
            validityRange = cms.EventRange(310001, 0, 1, 999999999, 0, 0)
        )
    ),
    subSystem = cms.untracked.string('TimingDiamond'),
    verbosity = cms.untracked.uint32(0)
)
