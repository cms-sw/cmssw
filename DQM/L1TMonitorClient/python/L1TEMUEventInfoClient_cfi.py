import FWCore.ParameterSet.Config as cms

l1temuEventInfoClient = cms.EDAnalyzer("L1TEMUEventInfoClient",
    prescaleLS = cms.untracked.int32(-1),
    monitorDir = cms.untracked.string(''),
    prescaleEvt = cms.untracked.int32(1),
    #masking sequence: dtf, dtp, ctf, ctp, rpc, gmt, etp, htp, rct, gct, gt
    maskedSystems = cms.untracked.vuint32(0,0,0,0,0,0,0,0,0,0,0),
    dataMaskedSystems = cms.untracked.vstring("all"),
    emulMaskedSystems = cms.untracked.vstring("empty")

)


