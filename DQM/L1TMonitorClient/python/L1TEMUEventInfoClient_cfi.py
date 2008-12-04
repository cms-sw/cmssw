import FWCore.ParameterSet.Config as cms

l1temuEventInfoClient = cms.EDFilter("L1TEMUEventInfoClient",
    prescaleLS = cms.untracked.int32(-1),
    monitorDir = cms.untracked.string(''),
    prescaleEvt = cms.untracked.int32(1),
    maskedSystems = cms.untracked.vuint32(0,1,1,0,0,0, 1,1,1,0,1)
    #masking sequence: dtf, dtp, ctf, ctp, rpc, gmt, etp, htp, rct, gct, gt

)


