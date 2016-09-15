import FWCore.ParameterSet.Config as cms

siStripBadModuleFedErrESSource = cms.ESSource("SiStripBadModuleFedErrESSource",
        appendToDataLabel = cms.string(''),
        ReadFromFile = cms.bool(True),
        FileName = cms.string('DQM.root'),
        BadStripCutoff = cms.double(0.8)
        )
