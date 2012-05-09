import FWCore.ParameterSet.Config as cms

# Entries for HZZ skim
#
# Dominique Fortin - UC Riverside
#
higgsToZZ4LeptonsFilter = cms.EDFilter("HiggsToZZ4LeptonsSkim",
    electronMinimumEt = cms.double(5.0),
    DebugHiggsToZZ4LeptonsSkim = cms.bool(False),
    ElectronCollectionLabel = cms.InputTag("pixelMatchGsfElectrons"),
    # Minimum number of identified leptons above pt threshold
    nLeptonMinimum = cms.int32(3),
    GlobalMuonCollectionLabel = cms.InputTag("globalMuons"),
    # Collection to be accessed
    RecoTrackLabel = cms.InputTag("recoTracks"),
    # Pt threshold for leptons
    muonMinimumPt = cms.double(5.0)
)


