import FWCore.ParameterSet.Config as cms

from CalibPPS.ESProducers.totemDAQMappingESSourceXML_cfi import totemDAQMappingESSourceXML as _xml

totemDAQMappingESSourceXML = _xml.clone(
    subSystem = "TotemT2",
    multipleChannelsPerPayload = cms.untracked.bool(False),
    configuration = cms.VPSet(
        #initial dummy diamond map copy
        cms.PSet(
            validityRange = cms.EventRange("1:min - 364982:max"),
            mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_totem_nt2_2021.xml"),
            maskFileNames = cms.vstring()
        ),
        #T2 firmware test files
        cms.PSet(
            validityRange = cms.EventRange("364983:min - 999999999:max"),
            mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_totem_nt2_2023.xml"),
            maskFileNames = cms.vstring()
        )
    ),
    sampicSubDetId = cms.uint32(6),
)
