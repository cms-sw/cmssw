import FWCore.ParameterSet.Config as cms

# SoftLeptonTagInfo producer for tagging caloJets with global muons 
softMuonTagInfos = cms.EDFilter("SoftLepton",
    refineJetAxis = cms.uint32(0), ## use calorimetric jet direction by default

    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    leptons = cms.InputTag("globalMuons"),
    leptonQualityCut = cms.double(0.5),
    jets = cms.InputTag("iterativeCone5CaloJets"),
    leptonDeltaRCut = cms.double(0.4), ## lepton distance from jet axis

    leptonChi2Cut = cms.double(9999.0) ## no cut on lepton's track's chi2/ndof

)


