import FWCore.ParameterSet.Config as cms


pfJetIDSelector = cms.PSet(
        version = cms.string('FIRSTDATA'),
        quality = cms.string('LOOSE'),
        NHF = cms.double(1.0),
        CHF = cms.double(0.0),
        NEF = cms.double(1.0),
        CEF = cms.double(1.0)
    )
