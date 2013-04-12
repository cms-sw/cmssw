import FWCore.ParameterSet.Config as cms
import RecoBTag.SoftLepton.muonSelection

# SoftLeptonTagInfo producer for tagging caloJets with dedicated "soft" electrons
softPFLeptonsTagInfo = cms.EDProducer("SoftPFLeptonTagInfoProducer",
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    jets = cms.InputTag("ak5PFJets"),
    MuonId =cms.int32(0),
    electrons = cms.InputTag("SPFElectrons"),
    muons     = cms.InputTag("SPFMuons"),	 
    SPFELabel= cms.string('SPFElectrons'),
    SPFMLabel= cms.string('SPFMuons')
)
