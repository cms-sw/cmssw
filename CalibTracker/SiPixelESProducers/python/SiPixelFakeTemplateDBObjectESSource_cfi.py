import FWCore.ParameterSet.Config as cms

SiPixelFakeTemplateDBObjectESSource = cms.ESSource("SiPixelFakeTemplateDBObjectESSource",
    templateIDs = cms.vstring("0010","0012")                            
)


