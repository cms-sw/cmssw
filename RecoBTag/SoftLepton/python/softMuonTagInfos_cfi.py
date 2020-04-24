import FWCore.ParameterSet.Config as cms
import RecoBTag.SoftLepton.muonSelection_cff as muonSelection

# SoftLeptonTagInfo producer for tagging caloJets with global muons
softMuonTagInfos = cms.EDProducer("SoftLepton",
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    jets = cms.InputTag("ak4PFJetsCHS"),
    leptons = cms.InputTag("muons"),
    leptonCands = cms.InputTag(""),         # optional
    leptonId = cms.InputTag(""),            # optional

    refineJetAxis = cms.uint32(0),          # use calorimetric jet direction by default

    leptonDeltaRCut = cms.double(0.4),      # lepton distance from jet axis
    leptonChi2Cut = cms.double(9999.0),     # no cut on lepton's track's chi2/ndof

    muonSelection = muonSelection.AllGlobalMuons
)
