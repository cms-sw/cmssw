import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
diMuonHistos = DQMEDAnalyzer('DiMuonHistograms',
                              MuonCollection = cms.InputTag("muons"),
                              VertexLabel     = cms.InputTag("offlinePrimaryVertices"),
                              BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),                              

                              etaBin = cms.int32(40),
                              etaBBin = cms.int32(40),
                              etaEBin = cms.int32(40),
                              
                              etaBMin = cms.double(0.),
                              etaBMax = cms.double(1.1),
                              etaECMin = cms.double(0.9),
                              etaECMax = cms.double(2.4),
                              
                              lowMassMin = cms.double(2.0),
                              lowMassMax = cms.double(12.0),
                              highMassMin = cms.double(70.0),
                              highMassMax = cms.double(110.0),
                              folder = cms.string("Muons/diMuonHistograms")
                              )
diMuonHistos_miniAOD = DQMEDAnalyzer('DiMuonHistograms',
                                      MuonCollection  = cms.InputTag("slimmedMuons"),
                                      VertexLabel     = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                      BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),                              
                                      
                                      etaBin = cms.int32(40),
                                      etaBBin = cms.int32(40),
                                      etaEBin = cms.int32(40),
                                      
                                      etaBMin = cms.double(0.),
                                      etaBMax = cms.double(1.1),
                                      etaECMin = cms.double(0.9),
                                      etaECMax = cms.double(2.4),
                                      
                                      lowMassMin = cms.double(2.0),
                                      lowMassMax = cms.double(12.0),
                                      highMassMin = cms.double(70.0),
                                      highMassMax = cms.double(110.0),
                                      folder = cms.string("Muons_miniAOD/diMuonHistograms")
                                      )

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
pA_2016.toModify(diMuonHistos, 
    etaBin = 350, 
    etaBBin = 350, 
    etaEBin = 350, 

    lowMassMin = 2.0, 
    lowMassMax = 51.0, 
    highMassMin = 55.0, 
    highMassMax = 125.0
    )
