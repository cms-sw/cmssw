import FWCore.ParameterSet.Config as cms

higgsToZZ4LeptonsSkimDiLeptonProducer = cms.EDProducer("HiggsToZZ4LeptonsSkimDiLeptonProducer",

    # Collection to be accessed
    diLeptonsOScoll = cms.InputTag("diLeptonsOS"),
    diLeptonsSScoll = cms.InputTag("diLeptonsSS"),
    diMuonsZcoll    = cms.InputTag("diMuonsZ"),
    diElectronsZcoll = cms.InputTag("diElectronsZ"),

    # Pt thresholds for leptons
    cutPt          = cms.double(10.0),
    cutEta         = cms.double(2.5)

)


