import FWCore.ParameterSet.Config as cms

l1ParamMuons = cms.EDProducer("FastL1MuonProducer",
    # Muons
    MUONS = cms.PSet(
        # The muon simtrack's must be taken from there
        simModule = cms.InputTag("famosSimHits","MuonSimTracks"),
        dtSimHits = cms.InputTag("MuonSimHits","MuonDTHits"),
        cscSimHits = cms.InputTag("MuonSimHits","MuonCSCHits"),
        rpcSimHits = cms.InputTag("MuonSimHits","MuonRPCHits")
        # Simulate  only simtracks in this eta range
#        MaxEta = cms.double(2.4),
#        MinEta = cms.double(-2.4)
    )
)


