import FWCore.ParameterSet.Config as cms

from CalibPPS.ESProducers.totemDAQMappingESSourceXML_cfi import totemDAQMappingESSourceXML as _xml

totemDAQMappingESSourceXML = _xml.clone(
    subSystem = "TotemT2",
    configuration = cms.VPSet(
        cms.PSet(
            validityRange = cms.EventRange("1:min - 999999999:max"),
            mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_totem_nt2_2021.xml"),
            maskFileNames = cms.vstring()
        )
    ),
    sampicSubDetId = cms.uint32(6),
)
