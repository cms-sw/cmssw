import FWCore.ParameterSet.Config as cms

TotemDAQMappingESSourceXML = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),

  mappingFileNames = cms.untracked.vstring(),
  maskFileNames = cms.untracked.vstring()
)
