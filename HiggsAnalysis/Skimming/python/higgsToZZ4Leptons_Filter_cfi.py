import FWCore.ParameterSet.Config as cms

# Entries for HZZ skim
#
# Dominique Fortin - UC Riverside
# modified by N. De Filippis - LLR 
#
higgsToZZ4LeptonsFilter = cms.EDFilter("HiggsToZZ4LeptonsSkim",
    DebugHiggsToZZ4LeptonsSkim = cms.bool(False),

    # Collection to be accessed
    ElectronCollectionLabel = cms.InputTag("pixelMatchGsfElectrons"),
    MuonCollectionLabel     = cms.InputTag("muons"),
  
    # Pt thresholds for leptons
    stiffMinimumPt          = cms.double(10.0),
    softMinimumPt           = cms.double(5.0),

    # Minimum number of identified leptons above pt threshold
    nStiffLeptonMinimum     = cms.int32(2),

    # nLepton is nSoft + nStiff
    nLeptonMinimum          = cms.int32(3),
)


