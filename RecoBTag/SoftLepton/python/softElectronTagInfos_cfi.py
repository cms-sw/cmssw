import FWCore.ParameterSet.Config as cms
import RecoBTag.SoftLepton.muonSelection

# SoftLeptonTagInfo producer for tagging caloJets with dedicated "soft" electrons
softElectronTagInfos = cms.EDProducer("SoftLepton",
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    jets = cms.InputTag("ak5CaloJets"),
    leptons = cms.InputTag("gsfElectrons"),
    leptonCands = cms.InputTag("softElectronCands"),
    leptonId = cms.InputTag(""),            # optional

    refineJetAxis = cms.uint32(0),          # use calorimetric jet direction minus electron momentum by default

    leptonDeltaRCut = cms.double(0.4),      # lepton distance from jet axis
    leptonChi2Cut = cms.double(10.0),       # no cut on lepton's track's chi2/ndof
    
    muonSelection = RecoBTag.SoftLepton.muonSelection.All   # not used
)
