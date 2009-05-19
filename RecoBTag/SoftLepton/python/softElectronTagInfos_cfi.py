import FWCore.ParameterSet.Config as cms
import RecoBTag.SoftLepton.muonSelection

# SoftLeptonTagInfo producer for tagging caloJets with dedicated "soft" electrons
softElectronTagInfos = cms.EDFilter("SoftLepton",
    jets = cms.InputTag("iterativeCone5CaloJets"),
    leptons = cms.InputTag("softElectronSelector"),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),

    refineJetAxis = cms.uint32(0),          # use calorimetric jet direction by default

    leptonQualityCut = cms.double(0.0),
    leptonDeltaRCut = cms.double(0.4),      # lepton distance from jet axis
    leptonChi2Cut = cms.double(9999.0),     # no cut on lepton's track's chi2/ndof
    
    muonSelection = RecoBTag.SoftLepton.muonSelection.All   # not used
)
