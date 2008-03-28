import FWCore.ParameterSet.Config as cms

# SoftLeptonTagInfo producer for tagging caloJets with dedicated "soft" electrons
softElectronTagInfos = cms.EDFilter("SoftLepton",
    refineJetAxis = cms.uint32(0), ## use calorimetric jet direction by default

    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    leptons = cms.InputTag("btagSoftElectrons"),
    leptonQualityCut = cms.double(0.0),
    jets = cms.InputTag("iterativeCone5CaloJets"),
    leptonDeltaRCut = cms.double(0.4), ## lepton distance from jet axis

    leptonChi2Cut = cms.double(9999.0) ## no cut on lepton's track's chi2/ndof

)


