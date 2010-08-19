import FWCore.ParameterSet.Config as cms
import RecoBTag.SoftLepton.muonSelection

# SoftLeptonTagInfo producer for tagging caloJets with global muons 
softMuonTagInfos = cms.EDProducer("SoftLepton",
    #Type of vertex.  Options are nominal(0,0,0), beamspot, or vertex
    vertexType = cms.string("vertex"),
    #InputTag for the vertex type above
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    jets = cms.InputTag("ak5CaloJets"),
    leptons = cms.InputTag("muons"),
    leptonCands = cms.InputTag(""),         # optional
    leptonId = cms.InputTag(""),            # optional
    
    refineJetAxis = cms.uint32(0),          # use calorimetric jet direction by default

    leptonDeltaRCut = cms.double(0.4),      # lepton distance from jet axis
    leptonChi2Cut = cms.double(9999.0),     # no cut on lepton's track's chi2/ndof
    
    muonSelection = RecoBTag.SoftLepton.muonSelection.AllGlobalMuons
)
